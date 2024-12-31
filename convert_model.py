#### convert llama/mixtral model for sparsity
import gc
import csv
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

def load_thresholds(threshold_path):
    """
    load thresholds from path
    """
    thresholds = torch.load(threshold_path)["up_proj_states_thresholds_2"]
    return thresholds

# up_th = None
up_th = load_thresholds('./saving/threshold/c4_mixtral_up/thresholds.pt')
# up_th = load_thresholds('./saving/threshold/chess/up_threshold/thresholds_0_7.pt')


class Linearlayer(nn.Module):
    """
    sparse linear layer
    """
    def __init__(self, weight, sparsity=0.1, num=15, name = None):
        super(Linearlayer, self).__init__()
        self.weight = weight.clone().to(weight.device)
        self.layer_idx = num

    def forward(self, x, indices=None, mask=None):
        if mask != None:
            ### Should multiply mask after x@w instead of modifying original weight with indices
            true_value = torch.mul(x@self.weight.T, mask)
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

        if self.num > start_num:
            # self.average_gate = torch.load(f'/mnt/newdata/lz/sparsity/c4_llama/new_channelgate/{num}-average.pth')
            # self.up_average = torch.load(f'/mnt/newdata/lz/sparsity/c4_llama/new_channelup/{num}-average.pth')
            ### saving for self.ratio average value
            # self.gate_threshold = gate_th[self.num].cuda()
            self.up_threshold = up_th[self.num].cuda()
            self.count_sum = 0
            self.token_sum = 0

        del module.gate_proj
        del module.up_proj

    def print_ratio(self):
        """
        print the average of self.ratio
        """ 
        print(f'layer {self.num} ratio: {self.count_sum/self.token_sum:.4f}')
        self.count_sum = 0
        self.token_sum = 0
                
    def forward(self, x):
        if self.num > self.start_num:
            ### Multiply complete up projection with gate average
            up_result = self.up_proj(x)
            # predicts = torch.abs(torch.mul(up_result, self.average_gate))
            gate_proj_states = self.act_fn(self.gate_proj(x))
            ### Threshold method
            up_proj_states = torch.where(up_result.abs() > self.up_threshold, up_result, 0.0, )
            # mask = (predicts >= gate_th[self.num]).to(x.dtype) ### 0 because there is only one expert for llama model
            
            ### Calculate actual preserved ratio
            # true_ratio = mask.float().sum().item()  # Calculate proportion of True values
            ### gate_proj_states 非0的个数
            true_ratio = (up_proj_states != 0).sum().item()
            self.count_sum += true_ratio
            self.token_sum += up_proj_states.numel()
            
            ### sparsity for up_mask_index
            true_value = up_proj_states * gate_proj_states         
            down_value = self.down_proj(true_value) ### No mask here to simplify computation since masked values are 0 anyway
        else:
            true_value = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_value = self.down_proj(true_value)

        return down_value
    
class MixtralLayer(BaseMLPLayer):
    """
    mlp Mixtral layer without coreinfer
    """
    def __init__(self, module, sparsity, layernum=15, expertnum=0, start_num=0, name = None, gamma=0.05):
        ### self.w1: gate, w3: up, w2: down
        super().__init__(layernum)
        self.layernum = layernum
        self.expertnum = expertnum

        self.act_fn = module.act_fn
        self.intermediate_size = module.w3.weight.size(0)  # 14336
        self.hidden_size = module.w3.weight.size(1)        # 4096
        self.sparsity = sparsity
        self.hc_nums = int(gamma * self.intermediate_size)
        self.start_num = start_num

        ### new layer converted from nn.linear
        self.gate_proj = Linearlayer(module.w1.weight, sparsity=sparsity, num=layernum, )
        self.up_proj = Linearlayer(module.w3.weight, sparsity=sparsity, num=layernum, )
        self.down_proj = module.w2

        if self.layernum > start_num:
            # self.average_gate = torch.load(f'/mnt/newdata/lz/sparsity/c4_llama/new_channelgate/{num}-average.pth')
            # self.up_average = torch.load(f'/mnt/newdata/lz/sparsity/c4_llama/new_channelup/{num}-average.pth')
            ### saving for self.ratio average value
            self.up_threshold = up_th[self.layernum][self.expertnum].to(module.w1.weight.device)
            self.count_sum = 0
            self.token_sum = 0

        del module.w1
        del module.w3

    def print_ratio(self):
        """
        print the average of self.ratio
        """ 
        print(f'layer {self.layernum} expert {self.expertnum} ratio: {self.count_sum/self.token_sum:.4f}')
        self.count_sum = 0
        self.token_sum = 0
                
    def forward(self, x):
        if self.num > self.start_num:
            ### Multiply complete up projection with gate average
            up_result = self.up_proj(x)
            # predicts = torch.abs(torch.mul(up_result, self.average_gate))
            gate_proj_states = self.act_fn(self.gate_proj(x))
            ### Threshold method
            up_proj_states = torch.where(up_result.abs() > self.up_threshold, up_result, 0.0, )
            # mask = (predicts >= gate_th[self.num]).to(x.dtype) ### 0 because there is only one expert for llama model
            
            ### Calculate actual preserved ratio
            # true_ratio = mask.float().sum().item()  # Calculate proportion of True values
            ### gate_proj_states 非0的个数
            true_ratio = (up_proj_states != 0).sum().item()
            self.count_sum += true_ratio
            self.token_sum += up_proj_states.numel()
            
            ### sparsity for up_mask_index
            true_value = up_proj_states * gate_proj_states         
            down_value = self.down_proj(true_value) ### No mask here to simplify computation since masked values are 0 anyway
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

def convert_mixtral_model(model, sparsity, start_num, end_num, token_sparsity=0.1, alpha=0.2, beta=0.1, gamma=0.3, sparsity_ratio=0, use_core=True):
    from tqdm import tqdm
    # global up_th
    # up_th = load_thresholds('./saving/threshold/c4_mixtral_up/thresholds.pt')
    for name, module in model.named_modules():
        if "experts" in name and name.count('.') == 5:
            # print(name)
            layer_num = int(name.split('.')[2])
            expert_num = int(name.split('.')[-1])
            # print(layer_num, expert_num)

            parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
            attr_name = name.rsplit('.', 1)[-1]
            if parent_name != '':
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model # for lm_head

            NewLayer = MixtralLayer(module, sparsity=0.1, layernum=layer_num, expertnum=expert_num, name = name, start_num=-1, gamma=0.3)
            setattr(parent, attr_name, NewLayer)
            del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model