#### convert llama model for sparsity
import gc
import torch 
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    """
    sparsity prediction model：
    input_dim: 4096
    output_dim: 14336
    lora hidden_dim: 1024
    """
    def __init__(self,input_dim,output_dim,hidden_dim=1024):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim,bias=False,dtype=torch.float16)
        self.linear2 = nn.Linear(hidden_dim,output_dim,bias=False,dtype=torch.float16)  

    def forward(self, x):
        return self.linear2(self.linear1(x))


class Linearlayer(nn.Module):
    """
    sparse linear layer
    """
    def __init__(self, weight, sparsity=0.1, num=15, name = None):
        super(Linearlayer, self).__init__()
        self.weight = weight.clone().cuda()
        intermediate_size = weight.size(0)
        neuron_num = int(intermediate_size * sparsity)
        self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).cuda()
        self.layer_idx = num

    def forward(self, x, indices=None):
        if indices != None:
            self.filtered_W = self.weight[indices,:]
            # print(self.filtered_W.shape)
            true_value = x@self.filtered_W.T
        else:
            true_value = x@self.weight.T
            
        return true_value

class Newlayer(Linearlayer):
    """
    sparse linear layer(only decode phase):
    in the prefill phase, the whole weight is used
    """
    def __init__(self, weight, sparsity=0.1, num=15, name = None):
        super(Newlayer, self).__init__(weight, sparsity, num, name)

    def forward(self, x, indices=None):
        if x.size(1) > 1:
            ### prefill
            true_value = x@self.weight.T.cuda()
        else:
            ### decode 
            if indices != None:
                self.filtered_W = self.weight[indices,:].clone().cuda()
                # print(self.filtered_W.shape)
                true_value = x@self.filtered_W.T.cuda()
            else:
                true_value = x@self.weight.T.cuda()
            
        return true_value
    
class BaseMLPLayer(nn.Module):
    """
    Base class for mlp layers with common methods
    """
    def __init__(self, num=15):
        super().__init__()
        self.num = num
        self.predict_all = []
        self.overlap_all = []
        self.overlap_clist = []

    def clear_list(self):
        self.predict_all = []
        self.overlap_all = []
        self.overlap_clist = []

    def coreinfer_recall(self):
        print(f'in decode, layer {self.num}')
        try:
            print(f'Predicted neuron num: {sum(self.predict_all)/len(self.predict_all):.1f}')
        except:
            print('No data')
        a_overlap_count = sum(self.overlap_clist) / len(self.overlap_clist)
        a_overlap_ratio = sum(self.overlap_clist) / sum(self.overlap_all)
        print(f'Overlap count: {a_overlap_count:.1f}, Overlap ratio: {a_overlap_ratio:.4f}')

class MLP_Core(BaseMLPLayer):
    """
    mlp layer with coreinfer
    in prefill phase, the whole weight is used and profile the mask index
    """
    def __init__(self, module, sparsity, num=15, name = None, alpha=0.3, beta=0.3, gamma=0.05):
        super().__init__(num)
        self.act_fn = module.act_fn
        self.intermediate_size = module.up_proj.weight.size(0)  # 14336
        self.hidden_size = module.up_proj.weight.size(1)        # 4096
        self.tc_nums = int(alpha * self.intermediate_size)
        self.sc_nums = int(beta * self.intermediate_size)
        self.x_topk = None
        self.sparsity = sparsity
        self.hc_nums = int(gamma * self.intermediate_size)
        neuron_num = int(self.intermediate_size * sparsity)

        ### new layer converted from nn.linear
        self.gate_proj = Newlayer(module.gate_proj.weight, sparsity=sparsity, num=num, )
        self.up_proj = Newlayer(module.up_proj.weight, sparsity=sparsity, num=num, )
        self.down_proj = module.down_proj
        self.filtered_W_down = torch.zeros((self.hidden_size, neuron_num)).to(torch.float16).cuda()

        if self.num > 0:
            self.helper = SimpleLinearModel(4096,14336,hidden_dim=1024).cuda()
            weight = torch.load(f'./output/sparsity/{self.num}-2.pt',map_location=module.down_proj.weight.device)
            self.helper.load_state_dict(weight)

        del module.gate_proj
        del module.up_proj

    def forward(self, x):
        ### todo: end-to-end inference
        if x.size(1)>1:
            ### prefill phase
            true_value = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            v = torch.abs(true_value)
            down_value = self.down_proj(true_value)

            mask_indices = torch.topk(v, self.tc_nums).indices.flatten()
            index_counts = torch.bincount(mask_indices, minlength=v.numel())
            self.x_topk = torch.topk(index_counts, self.sc_nums).indices
        else:
            ### decode phase
            global pre_x
            pre_x = x
            if self.num > 0:
                core_mask_index = self.x_topk
                up_mask_index = torch.topk(self.helper(pre_x), self.hc_nums).indices.flatten()
                combined_set = torch.cat((core_mask_index, up_mask_index))
                combine_mask_index = torch.unique(combined_set)
                ### 根据mask_index进行稀疏
                true_value = self.act_fn(self.gate_proj(x, combine_mask_index)) * self.up_proj(x, combine_mask_index)            
                self.filtered_W_down = self.down_proj.weight[:,combine_mask_index].clone().cuda()
                down_value = true_value @ self.filtered_W_down.T

                v = torch.abs(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                ### calculate recall
                self.predict_all.append(combine_mask_index.size(0))
                mask_indices = torch.topk(v, int(self.sparsity * self.intermediate_size)).indices.flatten()
                overlap_count = torch.isin(mask_indices, combine_mask_index).sum().item()
                self.overlap_clist.append(overlap_count)
                self.overlap_all.append(int(self.sparsity * self.intermediate_size))
            else:
                true_value = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                down_value = self.down_proj(true_value)

        return down_value
    
class MLPLayer(BaseMLPLayer):
    """
    mlp layer without coreinfer
    """
    def __init__(self, module, sparsity, num=15, start_num=0, name = None, gamma=0.05):
        super().__init__(num)
        self.act_fn = module.act_fn
        self.intermediate_size = module.up_proj.weight.size(0)  # 14336
        self.hidden_size = module.up_proj.weight.size(1)        # 4096
        self.sparsity = sparsity
        self.hc_nums = int(gamma * self.intermediate_size)
        self.start_num = start_num
        neuron_num = int(self.intermediate_size * sparsity)

        ### new layer converted from nn.linear
        self.gate_proj = Linearlayer(module.gate_proj.weight, sparsity=sparsity, num=num, )
        self.up_proj = Linearlayer(module.up_proj.weight, sparsity=sparsity, num=num, )
        self.down_proj = module.down_proj
        self.filtered_W_down = torch.zeros((self.hidden_size, neuron_num)).to(torch.float16).cuda()

        if self.num > start_num:
            self.helper = SimpleLinearModel(4096,14336,hidden_dim=1024).cuda()
            weight = torch.load(f'./output/sparsity/{self.num}-2.pt',map_location=module.down_proj.weight.device)
            self.helper.load_state_dict(weight)

        del module.gate_proj
        del module.up_proj
                
    def forward(self, x):
        global pre_x
        pre_x = x
        if self.num > self.start_num:
            ### the final token activation for topk-selection
            up_mask_index = torch.topk(self.helper(pre_x)[:,-1,:], self.hc_nums).indices.flatten()  # torch.Size([1, 300, 4300])
            # print(up_mask_index.size())
            ### sparsity for up_mask_index
            true_value = self.act_fn(self.gate_proj(x, up_mask_index)) * self.up_proj(x, up_mask_index)            
            self.filtered_W_down = self.down_proj.weight[:,up_mask_index]
            down_value = true_value @ self.filtered_W_down.T

            v = torch.abs(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            ### calculate recall
            self.predict_all.append(up_mask_index.size(0))
            mask_indices = torch.topk(v[:,-1,:], int(self.sparsity * self.intermediate_size)).indices.flatten()
            overlap_count = torch.isin(mask_indices, up_mask_index).sum().item()
            self.overlap_clist.append(overlap_count)
            self.overlap_all.append(int(self.sparsity * self.intermediate_size))
        else:
            true_value = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_value = self.down_proj(true_value)

        return down_value

def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity=0.1, alpha=0.2, beta=0.1, gamma=0.3, sparsity_ratio=0, use_core=True):
    from tqdm import tqdm
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        if "mlp" in name and name.count('.') == 3:
            # print(name)
            num = int(name.split('.')[2])
            if num>=start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                if use_core:
                    print(f"Converting Layer {name} with CoreInfer")
                    NewLayer = MLP_Core(module, sparsity, num=num, name = name, alpha=alpha, beta=beta, gamma=gamma, )
                else:
                    NewLayer = MLPLayer(module, sparsity, num=num, name = name, start_num=start_num, gamma=gamma)
                setattr(parent, attr_name, NewLayer)
                del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model