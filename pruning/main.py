import os
import torch
import argparse
import torch.nn as nn
import util.log as log
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pruning.function.helper as helper
from pruning.net.PruneVGG16 import PruneVGG16
from pruning.net.PruneLeNet5 import PruneLeNet5
import torch.optim.lr_scheduler as lr_scheduler
from pruning.net.PruneAlexNet import PruneAlexNet

parser = argparse.ArgumentParser()
parser.add_argument("net", help="Network name", type=str)  # LeNet, AlexNet VGG16
parser.add_argument("data", help="Dataset name", type=str)  # MNIST CIFAR10 CIFAR100
args = parser.parse_args()
if args.net:
    net_name = args.net
else:
    net_name = 'VGG16'
if args.data:
    dataset_name = args.data
else:
    dataset_name = 'CIFAR10'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
CUDA_LAUNCH_BLOCKING = 1 # 禁止并行
if use_cuda:
    print('Using CUDA')

net_and_data = net_name + '_' + dataset_name
train_epoch_list = {
    'LeNet_MNIST': 4,
    'AlexNet_CIFAR10': 28,
    'VGG16_CIFAR10': 60,
    'AlexNet_CIFAR100': 50,
    'VGG16_CIFAR100': 100,
}
train_epoch = train_epoch_list[net_and_data]

# Prune sensitivity
sensitivity_list = {
    'LeNet_MNIST': {
        'conv1': 0.39,
        'conv2': 0.8,
        'fc1': 1.01,
        'fc2': 0.76
    },
    'AlexNet_CIFAR10': {
        'conv1': 0.3,
        'conv': 0.49,
        'fc1': 1.3,
        'fc2': 1.3,
        'fc3': 0.4,
    },
    'VGG16_CIFAR10': {
        'conv1': 0.3,  # 90.18
        'conv': 0.55,
        'fc1': 2.02,  # 2.3
        'fc2': 2.02,
        'fc3': 0.6,
    },
    'AlexNet_CIFAR100': {
        'conv1': 0.3,
        'conv': 0.5,
        'fc1': 1.1,
        'fc2': 1.1,
        'fc3': 0.4,
    },
    'VGG16_CIFAR100': {
        'conv1': 0.3,
        'conv': 0.5,
        'fc1': 2.03,
        'fc2': 2.03,
        'fc3': 0.5,
    }
}
sensitivity = sensitivity_list[net_and_data]
print(sensitivity)

retrain_milestones_list = {
    'LeNet_MNIST': [
        []
    ],
    'AlexNet_CIFAR10': [
        [1], [], [2, 5], [6]
    ],
    'VGG16_CIFAR10': [
        [2], [8], [8], [8], [8],
        [8], [8], [8], [8], [8],
        [2, 8],
    ],
    'AlexNet_CIFAR100': [
        [5]
    ],
    'VGG16_CIFAR100': [
        [5, 10],
        [1, 6]
    ]
}
retrain_milestones = retrain_milestones_list[net_and_data]

# When accuracy in test dataset is more than max_accuracy, save the model
train_max_accuracy_list = {
    'LeNet_MNIST': 99.15,
    'AlexNet_CIFAR10': 89,
    'VGG16_CIFAR10': 90.3,
    'AlexNet_CIFAR100': 87.9,
    'VGG16_CIFAR100': 90
}
retrain_max_accuracy_list = {
    'LeNet_MNIST': 99.25,
    'AlexNet_CIFAR10': 88.81,
    'VGG16_CIFAR10': 90.27,
    'AlexNet_CIFAR100': 90,
    'VGG16_CIFAR100': 91
}
train_max_accuracy = train_max_accuracy_list[net_and_data]
retrain_max_accuracy = retrain_max_accuracy_list[net_and_data]

retrain_mode_list = {
    'LeNet_MNIST': [
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 6},
        {'mode': 'full', 'retrain_epoch': 10}
    ],
    'AlexNet_CIFAR10': [
        {'mode': 'fc', 'retrain_epoch': 5},
        {'mode': 'fc', 'retrain_epoch': 36},
        {'mode': 'conv', 'retrain_epoch': 10},
        {'mode': 'conv', 'retrain_epoch': 15},
    ],
    'VGG16_CIFAR10': [
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'fc', 'retrain_epoch': 10},
        {'mode': 'conv', 'retrain_epoch': 10},
        {'mode': 'conv', 'retrain_epoch': 10},
        {'mode': 'conv', 'retrain_epoch': 20},
    ],
    'AlexNet_CIFAR100': [
        {'mode': 'conv', 'retrain_epoch': 8},
        {'mode': 'conv', 'retrain_epoch': 8},
        {'mode': 'fc', 'retrain_epoch': 5},
        {'mode': 'fc', 'retrain_epoch': 5},
    ],
    'VGG16_CIFAR100': [
        {'mode': 'fc', 'retrain_epoch': 8},
        {'mode': 'fc', 'retrain_epoch': 8},
        {'mode': 'conv', 'retrain_epoch': 10},
    ]
}
retrain_mode_type = retrain_mode_list[net_and_data]
print(retrain_mode_type)

learning_rate_decay_list = {
    'LeNet': 1e-5,
    'AlexNet': 1e-3,
    'VGG16': 5e-4,
}
learning_rate_decay = learning_rate_decay_list[net_name]

prune_num_per_retrain = 3

train_batch_size_list = {
    'LeNet': 32,
    'AlexNet': 64,
    'VGG16': 128
}
train_batch_size = train_batch_size_list[net_name]
test_batch_size = 64

lr_list = {
    'LeNet_MNIST': 1e-2,
    'AlexNet_CIFAR10': 1e-2,
    'VGG16_CIFAR10': 1e-2,
    'AlexNet_CIFAR100': 1e-2,
    'VGG16_CIFAR100': 1e-2
}
lr = lr_list[net_and_data]

train_milestones_list = {
    'LeNet_MNIST': [],
    'AlexNet_CIFAR10': [20],
    'VGG16_CIFAR10': [32, 50],
    'AlexNet_CIFAR100': [16],
    'VGG16_CIFAR100': [32, 50]
}
train_milestones = train_milestones_list[net_and_data]

# After pruning, the LeNet_MNIST is retrained with 1/10 of the original network's learning rate
# After pruning, the AlexNet_CIFAR10 is retrained with 1/100 of the original network's learning rate
retrain_lr_list = {
    'LeNet': [
        lr / 1
    ],
    'AlexNet': [
        lr / 1
    ],
    'VGG16': [
        lr / 1,
    ]
}
retrain_lr = retrain_lr_list[net_name]

if net_name == 'LeNet':
    net = PruneLeNet5()
elif net_name == 'AlexNet':
    if dataset_name == 'CIFAR10':
        net = PruneAlexNet(num_classes=10)
    elif dataset_name == 'CIFAR100':
        net = PruneAlexNet(num_classes=100)
elif net_name == 'VGG16':
    if dataset_name == 'CIFAR10':
        net = PruneVGG16(num_classes=10)
    elif dataset_name == 'CIFAR100':
        net = PruneVGG16(num_classes=100)
else:
    net = None

path_root = './pruning/result/'
train_path = path_root + net_and_data + '.pth'
retrain_path = path_root + net_and_data + '_retrain.pth'
num_workers_list = {
    'LeNet': 10,
    'AlexNet': 10,
    'VGG16': 10
}
num_workers = num_workers_list[net_name]
trainloader, testloader = helper.load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers,
                                              name=dataset_name, net_name=net_name)

if not os.path.exists(path_root):
    os.mkdir(path_root)
# 定义损失函数、优化器、
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=learning_rate_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=train_milestones, gamma=0.1)

if use_cuda:
    # move param and buffer to GPU
    net = net.cuda()
    # speed up slightly
    cudnn.benchmark = True

# 只有CIFAR100数据集使用top5准确率，其余使用top1，只是说打印，返回的时候都是top1
if dataset_name == 'CIFAR100':
    top_5 = True
else:
    top_5 = False

# weight_decay is L2 regularization 
# 第一次train，前提得是无训练权重文件。auto_save设为false，不保存任何模型，have_save为false，再单独设置保存
if not os.path.exists(train_path):
    have_save = helper.train(testloader, net, trainloader, criterion, optimizer, train_path, scheduler,
                             train_max_accuracy,auto_save=False,
                             epoch=train_epoch, use_cuda=use_cuda, top_5=top_5)
    torch.save(net.state_dict(), train_path)

if use_cuda:
    net.load_state_dict(torch.load(train_path))
else:
    net.load_state_dict(torch.load(train_path, map_location='cpu'))

if net_name == 'LeNet':
    unit = 'K'
else:
    unit = 'M'

log.log_file_size(train_path, unit) # 打印单纯训练的文件大小
helper.test(use_cuda, testloader, net, top_5) # 测试单纯训练的top1准确度，并打印topk准确度

# Retrain 剪枝训练
for j in range(len(retrain_mode_type)): # lenet为6轮
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=retrain_lr[j] if j < len(retrain_lr) else retrain_lr[-1],
                          weight_decay=learning_rate_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=retrain_milestones[j] if j < len(retrain_milestones) else
                                         retrain_milestones[-1],
                                         gamma=0.1)
    retrain_mode = retrain_mode_type[j]['mode'] # 选择retrain模式，lenet全是full
    # net.prune_by_std(prune_mode=retrain_mode, use_cuda=use_cuda, sensitivity=sensitivity)
    net.prune_by_percent(0.09, len(retrain_mode_type), use_cuda=use_cuda) 
    '''百分比剪枝每层都剪；阈值剪枝只剪枝指定层'''
    if hasattr(net, 'drop_rate') and retrain_mode == 'fc':
        net.compute_dropout_rate()
    print('====================== After', j, 'Pruning ==================')
    helper.test(use_cuda, testloader, net, top_5)
    print('====================== Retrain', j, retrain_mode, 'Start ==================')
    if retrain_mode_type[j]['mode'] != 'full':
        net.fix_layer(net, fix_mode='conv' if retrain_mode == 'fc' else 'fc')
    
    have_save = helper.train(testloader, net, trainloader, criterion, optimizer, retrain_path, scheduler,
                             retrain_max_accuracy,unit, use_cuda=use_cuda, top_5=top_5,
                             epoch=retrain_mode_type[j]['retrain_epoch'], save_sparse=True)

    if not have_save: # have_save为false，单独设置保存稀疏模型
        helper.save_sparse_model(net, retrain_path, unit)