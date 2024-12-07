#### PD结合的思路去转换llama的模型
import gc
import torch 
import torch.nn as nn

# class DownLayer(nn.Module):
#     def __init__(self, weight, sparsity=0.1, num=15, alpha=0.1, beta=0.4, name = None):
#         super(DownLayer, self).__init__()
#         self.weight = weight.clone()
#         intermediate_size = weight.size(1)
#         self.neuron_num = int(intermediate_size*sparsity)
#         self.tc_nums = int(alpha * intermediate_size)
#         self.sc_nums = int(beta * intermediate_size)
        
#         # self.filtered_W = torch.zeros((weight.size(0),self.neuron_num)).to(torch.float16).cuda()
#         self.x_topk = None
#         self.layer_idx = num

#     def forward(self, x):
#         if x.size(1)>1:
#             self.weight_updated = False
#             x_train=x.clone()
#             true_value = x_train@self.weight.T.cuda()
#             true_value1 = x.squeeze().clone()

#             sorted_values, sorted_indices = torch.sort(true_value1, dim=1, descending=True)
#             limit=int(self.token_sparsity*(true_value1>0).sum().item()/true_value1.size(0))
#             top_indices = sorted_indices[:, :limit]
#             data_flattened = top_indices.reshape(-1)
#             unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
#             sorted_indices = torch.argsort(counts, descending=True)
#             sorted_indices_clu = unique_numbers[sorted_indices]

#             indices_all=sorted_indices_clu[:self.neuron_num].cpu()
#             if self.memory_limit:
#                 self.weight = self.weight.cpu()
#                 self.filtered_W = torch.zeros_like(self.weight).cuda().to(torch.float16)
                                                                         
#             self.filtered_W = self.weight[:,indices_all].clone().cuda()
            
#             global indices_list_all
#             if self.num ==6:
#                 indices_list_all=[]
                
#             indices_list_all.append(indices_all)

#             self.weight = self.weight.cpu()
#         else:
#             true_value = x @ self.filtered_W.T
            
#         return true_value

class UpLayer(nn.Module):
    def __init__(self, weight, sparsity=0.1, num=15, alpha=0.1, beta=0.4, name = None):
        super(UpLayer, self).__init__()
        self.weight = weight.clone().cuda()
        intermediate_size = weight.size(0)
        # print(intermediate_size, sparsity, intermediate_size * sparsity)
        self.neuron_num = int(intermediate_size * sparsity)
        self.tc_nums = int(alpha * intermediate_size)
        self.sc_nums = int(beta * intermediate_size)
        self.x_topk = None
        self.layer_idx = num

    def coreinfer_recall(self):
        print(f'in decode, up layer {self.layer_idx}')
        a_overlap_count = sum(self.overlap_clist) / len(self.overlap_clist)
        a_overlap_ratio = sum(self.overlap_clist) / sum(self.overlap_all)
        print(f'Overlap count: {a_overlap_count:.4f}, Overlap ratio: {a_overlap_ratio:.4f}')

    def forward(self, x):
        true_value = x@self.weight.T.cuda()
        if x.size(1)>1:
            print('[prefill] in up layer:', self.layer_idx)
            ### prefill
            mask_indices = torch.topk(true_value, self.tc_nums).indices.flatten()
            index_counts = torch.bincount(mask_indices, minlength=true_value.numel())
            self.x_topk = torch.topk(index_counts, self.sc_nums).indices
            self.overlap_all = []
            self.overlap_clist = []
        else:
            ### decode 
            # print(true_value[0][0].shape, self.neuron_num)
            mask_indices = torch.topk(true_value, self.neuron_num).indices.flatten()
            overlap_count = torch.isin(mask_indices, self.x_topk).sum().item()
            self.overlap_clist.append(overlap_count)
            self.overlap_all.append(mask_indices.numel())
        return true_value, mask_indices

class GateLayer(nn.Module):
    def __init__(self, weight, sparsity=0.1, num=15, alpha=0.1, beta=0.4, name = None):
        super(GateLayer, self).__init__()
        self.weight = weight.clone().cuda()
        intermediate_size = weight.size(0)
        # print(intermediate_size, sparsity, intermediate_size * sparsity)
        self.neuron_num = int(intermediate_size * sparsity)
        self.tc_nums = int(alpha * intermediate_size)
        self.sc_nums = int(beta * intermediate_size)
        self.x_topk = None
        self.layer_idx = num

    def coreinfer_recall(self):
        print(f'in decode, gate layer {self.layer_idx}')
        a_overlap_count = sum(self.overlap_clist) / len(self.overlap_clist)
        a_overlap_ratio = sum(self.overlap_clist) / sum(self.overlap_all)
        print(f'Overlap count: {a_overlap_count:.4f}, Overlap ratio: {a_overlap_ratio:.4f}')

    def forward(self, x):
        true_value = x@self.weight.T.cuda()
        if x.size(1)>1:
            print('[prefill] in gate layer:', self.layer_idx)
            ### prefill 
            mask_indices = torch.topk(true_value, self.tc_nums).indices.flatten()
            index_counts = torch.bincount(mask_indices, minlength=true_value.numel())
            self.x_topk = torch.topk(index_counts, self.sc_nums).indices
            ### Calculate coverage
            self.overlap_all = []
            self.overlap_clist = []
        else:
            ### decode 
            # print(true_value[0][0].shape, self.neuron_num)
            mask_indices = torch.topk(true_value, self.neuron_num).indices.flatten()
            overlap_count = torch.isin(mask_indices, self.x_topk).sum().item()
            self.overlap_clist.append(overlap_count)
            self.overlap_all.append(mask_indices.numel())
        return true_value


class MLPLayer(nn.Module):
    def __init__(self, module, sparsity, num=15, name = None, alpha=0.3, beta=0.3):
        super().__init__()
        # print(module)
        # self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
        self.layer_idx = num
        self.beta = beta
        self.intermediate_size = 14336

        #### load the up predictor(wanda pruning)
        sparsity_ratio = 0
        self.helper = torch.load(f'/home/lz/workspace/llama2-7b/moe-offloading/notebooks/output/sparsity/wanda/up_proj_{sparsity_ratio}.pt')

        self.overlap_all = []
        self.overlap_clist = []

        # self.up_proj = UpLayer(module.up_proj.weight, sparsity, num=num, name = name, alpha=alpha, beta=beta)
        # del module.up_proj
        self.gate_proj = GateLayer(module.gate_proj.weight, sparsity, num=num, name = name, alpha=alpha, beta=beta)
        del module.gate_proj

    def coreinfer_recall(self):
        print(f'in decode, layer {self.layer_idx}')
        a_overlap_count = sum(self.overlap_clist) / len(self.overlap_clist)
        a_overlap_ratio = sum(self.overlap_clist) / sum(self.overlap_all)
        print(f'Overlap count: {a_overlap_count:.4f}, Overlap ratio: {a_overlap_ratio:.4f}')
                
    def forward(self, x):
        true_gate = self.gate_proj(x)
        true_up = self.up_proj(x)
        true_value = self.act_fn(true_gate) * true_up
        ### todo: test for upper bound 
        if x.size(1)>1:
            ### prefill phase
            ### Calculate coverage
            self.overlap_all = []
            self.overlap_clist = []
        else:
            ### decode phase
            # print(true_value[0][0].shape, self.neuron_num)
            gate_mask_index = self.gate_proj.x_topk
            up_mask_index = torch.topk(x @ self.helper.T, int(self.beta * self.intermediate_size)).indices

            ## torch取两个index集合的交集
            combine_mask_index = gate_mask_index[torch.isin(gate_mask_index, up_mask_index)]
            print('gate, up, combine 的维度:',gate_mask_index.shape, up_mask_index.shape, combine_mask_index.shape)
            mask_indices = torch.topk(true_value, int(0.1 * self.intermediate_size)).indices.flatten()
            overlap_count = torch.isin(mask_indices, combine_mask_index).sum().item()
            self.overlap_clist.append(overlap_count)
            self.overlap_all.append(mask_indices.numel())

        down_proj = self.down_proj(true_value)

        return down_proj


def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity=0.1, alpha=0.3, beta=0.3):
    from tqdm import tqdm
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        if "mlp" in name and name.count('.') == 3:
            # print(name)
            num = int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                NewLayer = MLPLayer(module, sparsity, num=num, name = name, alpha=alpha, beta=beta)
                setattr(parent, attr_name, NewLayer)
                del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model