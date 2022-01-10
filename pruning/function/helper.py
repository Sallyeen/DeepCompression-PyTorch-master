import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import util.log as log
from scipy.sparse import csr_matrix
import torchvision.transforms as transforms


# 读取数据并做一些变换，返回train和test的dataloader
def load_dataset(use_cuda, train_batch_size, test_batch_size, num_workers, name='MNIST', net_name='LeNet',
                 data_dir='./data'):
    train_set = None
    test_set = None
    transform_train = None
    transform_test = None
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    # num_workers指工作进程，每个worker会分到指定batch，加载到内存
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]) # transform做图像变换，这里为【化为张量，正则化】
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True,
                                               download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False,
                                              download=True, transform=transform)
    elif name == 'CIFAR10':
        if net_name == 'VGG16':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) # train数据集，这里为【随机裁剪、水平翻转、化为张量，正则化】
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) # test数据集，这里为【化为张量，正则化】
        elif net_name == 'AlexNet':
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) # train数据集，这里为【resize、水平翻转、化为张量，正则化】

            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) # test数据集，这里为【resize、化为张量，正则化】
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                 download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                                download=True, transform=transform_test)
    elif name == 'CIFAR100':
        if net_name == 'VGG16':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]) # train数据集，这里为【随机裁剪、水平翻转、化为张量，正则化】
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]) # 数据集，这里为【化为张量，正则化】
        elif net_name == 'AlexNet':
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]) # train数据集，这里为【resize、水平翻转、化为张量，正则化】
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]) # test数据集，这里为【resize、化为张量，正则化】
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                  download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                                 download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, **kwargs)
    return trainloader, testloader

# 计算topk准确个数:outputs [64, 10], pred [64, 5], labels [64]
def top_k_accuracy(outputs, labels, topk=(1,)):
    maxk = max(topk) # 取top1或top5准确率专用写法，此处用到的是top5
    _, pred = outputs.topk(maxk, 1, True, True) # 沿着第一维排序后的最大值；分别返回最大值和索引
    pred = pred.t().type_as(labels) # t()表示转置；type_as()表示类型转换为传入参数的类型-->pred[5, 64]
    correct = pred.eq(labels.view(1, -1).expand_as(pred))# .eq()表示两个张量比较，值一样对应位置返回true
    # view(1, -1)表示把第零维变成1，expand_as表示扩为pred的形状，必须是扩张--->labels[5, 64],correct[5, 64]
    res = []
    for k in topk: # 对于top1与top5两种准确率
        correct_k = correct[:k].contiguous().view(-1).float().sum(0).item() # 对于correct[5, 64],64个目标
        # 有五种预测结果，最多只可能有一个true，所以correct_k <= 64
        res.append(correct_k) # res[top1的correct_k， top5的correct_k]
    return res

# 测试过程，打印准确度，并返回top_1准确度分数
def test(use_cuda, testloader, net, top_5=False):
    correct_1 = 0
    correct_5 = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            corr = top_k_accuracy(outputs, labels, topk=(1, 5)) # corr为n个样本中预测准确的个数
            total += labels.size(0) # 样本总个数
            correct_1 += corr[0] # 提取top1的预测准确个数
            correct_5 += corr[1] # 提取top5的预测准确个数
    top_1_accuracy = (100 * correct_1 / total) # top1预测准确分数
    top_5_accuracy = (100 * correct_5 / total) # top5预测准确分数
    if top_5:
        print('%.2f' % top_1_accuracy, '%.2f' % top_5_accuracy) # 如果计算top5，两个准确分数都打印
    else:
        print('%.2f' % top_1_accuracy) # 如果计算top1，打印top1预测准确分数
        # print('Accuracy of the network on the test images: %.2f %%' % accuracy)
    return top_1_accuracy # whatever， 返回top1预测准确分数

# 训练过程，执行一次就训练全部epoch
def train(testloader, net, trainloader, criterion, optimizer, train_path, scheduler, max_accuracy, unit='K',
          save_sparse=False, epoch=1, use_cuda=True, auto_save=True, top_5=False):
    have_save = False
    for epoch in range(epoch):
        # adjust_learning_rate(optimizer, epoch)
        train_loss = []
        # valid_loss = []
        net.train()
        # for inputs, labels in tqdm(trainloader):
        i = 0
        for inputs, labels in trainloader:
            # get the inputs
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)  # forward
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backward
            optimizer.step()  # update weight， 模型更新
            train_loss.append(loss.item())

        with torch.no_grad(): # 表示在这个语句块里，都不进行梯度更新
            # mean_train_loss = np.mean(train_loss)
            # print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
            # "Valid Loss: %5f" % mean_valid_loss
            accuracy = test(use_cuda, testloader, net, top_5)
            '''if top_5:
                print('train top_1_accuracy: {:.2f}%'.format(score[0]), 'train top_5_accuracy: {:.2%}'.format(score[1]))
            else:
                print('train top_1_accuracy: {:.2f}%'.format(score[0]))'''
            scheduler.step()
            if auto_save and accuracy > max_accuracy: # auto_save默认true,并且准确度符合要求
                if save_sparse: # 决定是否保存稀疏模型
                    save_sparse_model(net, train_path, unit)
                else:
                    torch.save(net.state_dict(), train_path)
                max_accuracy = accuracy
                have_save = True
    return have_save


# 没看懂：需要索引差，而非绝对位置，需要比边界值更大的索引差时，填充零防止溢出
def filler_zero(value, index, max_bits): # 传入参数为稀疏矩阵的值、索引、max_bits256；索引为元组序列的形式
    last_index = -1
    max_bits_minus = max_bits - 1 # 255/15
    i = 0

    if index.size == 0: # 都剪没了的情况
        return index, index

    # Save filler zero num
    filler_num_array = []
    # Save filler zero position index
    filler_index_array = []
    # print(len(index)) 308| 12 15445 30 247120 308 3089 6
    while i < len(index): # 循环的目的 一是更新索引，变为距离；二是记录需要补零的个数
        diff = index[i] - last_index - 1 # 刚开始diff为第一个非零元素的索引；
        if diff > max_bits_minus:
            filler_num = math.floor(diff / max_bits)
            filler_num_array.append(filler_num)
            filler_index_array.append(i)
            last_index += filler_num * max_bits
        else:
            last_index = index[i] # 更新上一个记录索引
            index[i] = diff
            i += 1

    new_len = value.size + sum(filler_num_array) # 因为距离超限的要补零，加号右边为补零的总个数
    new_value = np.empty(new_len, dtype=np.float32)
    new_index = np.empty(new_len, dtype=np.uint16) # 重新创建 索引与值
    
    k = 0 # index of new_index and new_value，最多 new_len
    j = 0 # index of filler_index_array and filler_num_array，最多 len(filler_index_array)
    n = 0 # index of index and value，最多 len(index)
    while k < new_len:
        if j < len(filler_index_array) and filler_index_array[j] == n:
            filler_num = filler_num_array[j]
            for m in range(filler_num):
                new_index[k] = max_bits_minus
                new_value[k] = 0
                k += 1
            j += 1
        else:
            new_index[k] = index[n]
            new_value[k] = value[n]
            n += 1
            k += 1

    return new_value, new_index

# 没看懂：保存剪枝后稀疏模型，
def save_sparse_model(net, path, unit):
    nz_num = [] 
    conv_diff_array = []
    fc_diff_array = []
    conv_value_array = []
    fc_value_array = []
    for key, tensor in net.state_dict().items():
        # print(key, tensor)：每个权重层分别有四个
        if key.endswith('mask'):
            continue
        if key.startswith('conv'):
            # 8 bits for conv layer index diff
            # print("tensor is:", key, tensor)            
            mat = csr_matrix(tensor.cpu().reshape(-1)) # tensor是个稀疏矩阵，仅保存【非零元素的索引与值】
            # print("csr_matrix is:", mat)  (0, 4) -0.059824314
            # print("csr_matrix.data is:", mat.data) [-0.06467187 -0.05982431]
            # print("csr_matrix.indices is:", mat.indices)  [1  4  5  9]
            bits = 8
            max_bits = 2 ** bits # 卷积层，2 ^ 8 = 256
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits) # 调用填充零方法
            # 返回零填充之后的new_arrar/new_key
            conv_diff_array.extend(diff_list)
            conv_value_array.extend(value_list)
        else:
            # 4 bits for fc layer index diff
            mat = csr_matrix(tensor.cpu().reshape(-1))
            bits = 4
            max_bits = 2 ** bits # 全连接层，2 ^ 4 = 16
            value_list, diff_list = filler_zero(mat.data, mat.indices, max_bits)
            fc_diff_array.extend(diff_list) # 全连接层填充零之后的索引
            fc_value_array.extend(value_list) # 全连接层填充零之后的值
        # print(len(diff_list)) 308 12 15445 30 247121 308 3089 6
        # print('diff_list:', diff_list)
        # print('fc_value_array:', fc_value_array) [-0.027408628, 0.059202306]
        nz_num.append(len(diff_list)) # nz_num保存权重层零填充之后的稀疏矩阵元素总个数

    length = len(fc_diff_array) # 全连接层在零填充之后的稀疏矩阵元素总个数
    # print('len(fc_diff_array):', len(fc_diff_array)) 250524
    if length % 2 != 0: # 若全连接层元素个数是奇数，则凑成偶数
        fc_diff_array.append(0)

    fc_diff_array = np.asarray(fc_diff_array, dtype=np.uint8)
    fc_merge_diff = []
    for i in range(int((len(fc_diff_array)) / 2)): # 循环1/2个全连接层元素个数125262；寻找两个相连的零
        fc_merge_diff.append((fc_diff_array[2 * i] << 4) | fc_diff_array[2 * i + 1])

    nz_num = np.asarray(nz_num, dtype=np.uint32)
    layer_nz_num = nz_num[0::2] + nz_num[1::2] # 换顺序，先权重层再全连接层
    if unit == 'K':
        temp = 1024
    else:
        temp = 1048576
    print('The parameters are', round(nz_num.sum() / temp, 2), unit, layer_nz_num) # 表示未剪枝的参数个数
    conv_diff_array = np.asarray(conv_diff_array, dtype=np.uint8)
    fc_merge_diff = np.asarray(fc_merge_diff, dtype=np.uint8)
    conv_value_array = np.asarray(conv_value_array, dtype=np.float32)
    fc_value_array = np.asarray(fc_value_array, dtype=np.float32)
    # print('fc_value_array:', fc_value_array) [-0.02679016  0.02325448  0.0142545  
    # ... -0.03383125  0.06480158  -0.06312249]

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    conv_value_array.dtype = np.uint8
    fc_value_array.dtype = np.uint8
    # print(conv_value_array)

    sparse_obj = np.concatenate((nz_num, conv_diff_array, fc_merge_diff, conv_value_array, fc_value_array))
    sparse_obj.tofile(path)
    log.log_file_size(path, unit)
