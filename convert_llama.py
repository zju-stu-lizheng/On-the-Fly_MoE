#### PD结合的思路去转换llama的模型
import gc
import torch 
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=32):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim,bias=False,dtype=torch.float16)
        # self.activation = nn.SiLU() # 添加激活函数
        self.linear2 = nn.Linear(hidden_dim,output_dim,bias=False,dtype=torch.float16)  
        # init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # print(x, self.linear1.weight)
        return self.linear2(self.linear1(x))

class MLPLayer(nn.Module):
    def __init__(self, module, sparsity, num=15, name = None, alpha=0.3, beta=0.3, gamma=0.05, sparsity_ratio=0):
        super().__init__()
        # print(module)
        self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
        self.num = num
        self.intermediate_size = module.up_proj.weight.size(0)
        self.tc_nums = int(alpha * self.intermediate_size)
        self.sc_nums = int(beta * self.intermediate_size)
        self.x_topk = None
        self.sparsity = sparsity

        #### load the up predictor(wanda pruning)
        # self.helper = torch.load(f'/home/lz/workspace/llama2-7b/moe-offloading/notebooks/output/sparsity/wanda/up_proj_{sparsity_ratio}.pt')
        if self.num == 15:
            self.helper = SimpleLinearModel(4096,14336,hidden_dim=1024).cuda()
            weight = torch.load(f'./output/sparsity/{self.num}-2.pt',map_location=module.gate_proj.weight.device)
            self.helper.load_state_dict(weight)
        self.hc_nums = int(gamma * self.intermediate_size)

        self.predict_all = []
        self.overlap_all = []
        self.overlap_clist = []

    def clear_list(self):
        self.predict_all = []
        self.overlap_all = []
        self.overlap_clist = []

    def coreinfer_recall(self):
        print(f'in decode, layer {self.num}')
        print(f'Predicted neuron num: {sum(self.predict_all)/len(self.predict_all):.1f}')
        a_overlap_count = sum(self.overlap_clist) / len(self.overlap_clist)
        a_overlap_ratio = sum(self.overlap_clist) / sum(self.overlap_all)
        print(f'Overlap count: {a_overlap_count:.1f}, Overlap ratio: {a_overlap_ratio:.4f}')
                
    def forward(self, x):
        true_gate = self.gate_proj(x)
        true_up = self.up_proj(x)
        true_value = self.act_fn(true_gate) * true_up
        v = torch.abs(true_value)
        ### todo: test for upper bound 
        if x.size(1)>1:
            ### prefill phase
            mask_indices = torch.topk(v, self.tc_nums).indices.flatten()
            index_counts = torch.bincount(mask_indices, minlength=v.numel())
            self.x_topk = torch.topk(index_counts, self.sc_nums).indices
            ### Calculate coverage
            # self.overlap_all = []
            # self.overlap_clist = []
            # self.predict_all = []
        else:
            ### decode phase
            if self.num == 14:
                global pre_x
                pre_x = x
                # print(x)
            # print(true_value[0][0].shape, self.neuron_num)
            elif self.num == 15:
                core_mask_index = self.x_topk
                # print(pre_x)
                up_mask_index = torch.topk(self.helper(pre_x), self.hc_nums).indices.flatten()

                ## torch取两个index集合的交集
                # combine_mask_index = core_mask_index[torch.isin(core_mask_index, up_mask_index)]
                ## torch取两个index集合的并集
                # 拼接两个索引集合
                combined_set = torch.cat((core_mask_index, up_mask_index))
                # 使用 torch.unique 去除重复元素
                combine_mask_index = torch.unique(combined_set)

                # print('The dimension of combine:', combine_mask_index.size(0), f'{combine_mask_index.size(0)/self.intermediate_size:.4f}')
                self.predict_all.append(combine_mask_index.size(0))
                mask_indices = torch.topk(v, int(self.sparsity * self.intermediate_size)).indices.flatten()
                # print(f"core infer :{torch.isin(mask_indices, core_mask_index).sum().item()/mask_indices.numel():.4f}", )
                overlap_count = torch.isin(mask_indices, combine_mask_index).sum().item()
                self.overlap_clist.append(overlap_count)
                self.overlap_all.append(int(self.sparsity * self.intermediate_size))

        down_proj = self.down_proj(true_value)

        return down_proj

def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity=0.1, alpha=0.3, beta=0.3, gamma=0.05, sparsity_ratio=0):
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
                NewLayer = MLPLayer(module, sparsity, num=num, name = name, alpha=alpha, beta=beta, gamma=gamma, sparsity_ratio=sparsity_ratio)
                setattr(parent, attr_name, NewLayer)
                del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model