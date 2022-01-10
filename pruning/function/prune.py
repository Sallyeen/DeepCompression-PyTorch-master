import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class MaskModule(Module):
    # 阈值剪枝：小于阈值的置零
    def prune_theshold(self, threshold, use_cuda=True, bias_threshold=None):
        zero_weight = torch.zeros_like(self.weight.data).float()
        if use_cuda:
            zero_weight = zero_weight.cuda()
        # Set the weight less than threshold to zero
        new_weight_mask = torch.where(abs(self.weight.data) < threshold, zero_weight, self.weight_mask.float())
        self.weight.data = self.weight * new_weight_mask.float()
        self.weight_mask.data = new_weight_mask
        if self.bias is not None and bias_threshold is not None:
            zero_bias = torch.zeros_like(self.bias.data).float()
            if use_cuda:
                zero_bias = zero_bias.cuda()
            # Set the bias less than bias_threshold to zero
            new_bias_mask = torch.where(abs(self.bias.data) < bias_threshold, zero_bias, self.bias_mask.float())
            self.bias.data = self.bias * new_bias_mask.float()
            self.bias_mask.data = new_bias_mask

    # 按百分数剪枝：用百分数计算阈值，再采用阈值剪枝
    def prune_by_percent_once(self, percent, use_cuda):
        # Put the weights that aren't masked out in sorted order.weight张量内的值按升序排列
        sorted_weights = torch.sort(torch.abs(self.weight.data[self.weight_mask.data == 1]))

        # Determine the cutoff for weights to be pruned.确定权重剪枝个数，与阈值
        cutoff_index = math.ceil(percent * torch.numel(sorted_weights[0]))
        cutoff = (sorted_weights[0][cutoff_index]).item() # 权重层要被剪枝的阈值【小的中的最大的】
        sorted_bias = torch.sort(torch.abs(self.bias.data[self.bias_mask.data == 1]))
        bias_cutoff_index = math.ceil(percent * torch.numel(sorted_bias[0]))
        if len(sorted_bias[0]) > 0 and bias_cutoff_index < len(sorted_bias[0]):
            bias_cutoff = (sorted_bias[0][bias_cutoff_index]).item() # bias要被剪枝的阈值【小的中的最大的】
        else:
            bias_cutoff = 0
        # Prune all weights below the cutoff.调用阈值剪枝协助剪枝
        self.prune_theshold(cutoff, use_cuda, bias_cutoff)

# 用于全连接层【线性】的掩码操作
class MaskLinearModule(MaskModule):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        # Add mask for weight，初始权重无值，初始权重掩码全一
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
            # Add mask for bias
            self.bias_mask = nn.Parameter(torch.ones(out_features).byte(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        weight = self.weight * self.weight_mask.float()
        if self.bias is not None:
            return F.linear(input, weight, self.bias * self.bias_mask.float())
        else:
            return F.linear(input, weight)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

# 用于卷积层的剪枝的掩码操作
class MaskConv2Module(MaskModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *(kernel_size, kernel_size)),
                                   requires_grad=True)
        # Add mask for weight
        self.weight_mask = nn.Parameter(torch.ones(self.weight.shape).byte(), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
            # Add mask for bias
            self.bias_mask = nn.Parameter(torch.ones(out_channels).byte(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        weight = self.weight * self.weight_mask.float()
        if self.bias is not None:
            bias = self.bias * self.bias_mask.float()
            return F.conv2d(input, weight, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(input, weight, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)

    def reset_parameters(self): # 权重与bias初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class PruneModule(Module):
    @staticmethod # 返回某一层mask.data的未剪枝元素个数与元素总个数
    def compute_prune_num(layer, is_bias=False):
        if is_bias:
            array = layer.bias_mask.data
        else:
            array = layer.weight_mask.data
        unpruned_num = int(torch.sum(array))
        total_num = int(torch.numel(array))
        return unpruned_num, total_num

    # 在main函数中被call，循环更新drop_rate数值
    def compute_dropout_rate(self):
        for index in range(len(self.drop_rate)):
            # Last Layer返回某一fc层权重mask.data的未剪枝元素个数与元素总个数
            last_unpruned_num, last_total_num = self.compute_prune_num(self.fc_list[index])
            # Next Layer返回下一fc层权重mask.data的未剪枝元素个数与元素总个数
            next_unpruned_num, next_total_num = self.compute_prune_num(self.fc_list[index + 1])

            # If define as
            # p = 0.5 * math.sqrt(last_not_prune_num * next_not_prune_num / last_total_num * next_total_num)
            # the result of multiplication maybe overflow
            p = 0.5 * math.sqrt((last_unpruned_num / last_total_num) * (next_unpruned_num / next_total_num))
            # print('The drop out rate is:', round(p, 5))
            self.drop_rate[index] = p # 更新drop_rate数值

    # 标准差阈值剪枝：计算每一个符合要求的modules阈值，并调用阈值剪枝
    def prune_by_std(self, sensitivity=None, use_cuda=True, prune_mode='full'):
        prune_mode_list = ['full', 'conv', 'fc']
        if prune_mode not in prune_mode_list:
            return
        if sensitivity is None:
            sensitivity = {
                'fc': 0.77,
                'conv1': 0.3,
                'conv': 0.5,
            }
        # print('===== prune', prune_mode, '=====')
        for name, module in self.named_modules():
            if name != '' and (prune_mode == 'full' or name.startswith(prune_mode)):
                # The pruning threshold is chosen as a quality parameter multiplied
                # by the standard deviation of a layer's weight
                if name in sensitivity:
                    s = sensitivity[name]
                elif name.startswith('fc'):
                    s = sensitivity['fc']
                else:
                    s = sensitivity['conv']
                #  The pruning threshold is chosen
                # as a quality parameter s multiplied by the standard deviation of a layer’s weights
                filter_weight = torch.masked_select(module.weight, module.weight_mask.byte())
                threshold = torch.std(filter_weight) * s # 阈值为 张量的标准差 × sensitivity
                filter_bias = torch.masked_select(module.bias, module.bias_mask.byte())
                bias_threshold = torch.std(filter_bias) * s
                module.prune_theshold(threshold, use_cuda, bias_threshold)

    # 百分比剪枝：对modules的每一层，只要name不为空[单层]，就call百分比修剪一次
    def prune_by_percent(self, percents, prune_num, use_cuda):
        for i, (name, module) in enumerate(self.named_modules()):
            if name != '':
                module.prune_by_percent_once(1 - math.pow(percents, 1 / prune_num), use_cuda)
            # print(f"name: {name}, modules: {module}")

    # 不管mask与bn，让fix_mode剪枝的停更梯度,也就是参与剪枝的不再需要梯度进行反向传播，其余需要梯度requires_grad
    def fix_layer(self, net, fix_mode='not'):
        if fix_mode == 'not':
            return
        # print('===== fix mode is', fix_mode, '=====')
        for name, p in net.named_parameters():
            if name.endswith('mask') or name.startswith('bn'):
                continue
            elif name.startswith(fix_mode):
                p.requires_grad = False
            else:
                p.requires_grad = True
            # print(f"name: {name}, param: {p}")

