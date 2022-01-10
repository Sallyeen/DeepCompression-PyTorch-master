import torch.nn.functional as F
import pruning.function.prune as prune


class PruneLeNet5(prune.PruneModule):
    def __init__(self):
        super(PruneLeNet5, self).__init__()
        self.conv1 = prune.MaskConv2Module(1, 20, 5, 1) # in_channels/out_channels/kernel_size/stride
        self.conv2 = prune.MaskConv2Module(20, 50, 5, 1)
        self.fc1 = prune.MaskLinearModule(4 * 4 * 50, 500)
        self.fc2 = prune.MaskLinearModule(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # 返回输入值x的第零维外维度乘积。例如x[m,n,q,w]，返回n*q*w
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

