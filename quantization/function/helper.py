import math
import time
import torch
import numpy as np
from tqdm import tqdm
from pruning.function.helper import test

# 1/8 加载剪枝后的稀疏模型，其中全连接层索引要加倍
def load_sparse_model(net, path, fc_bits):
    '''load the model which is saved as sparse matrix

    Args:
        net:  the network object
        path: the path of the pruned model
        bits: the bits of each index in fc layer

    Returns:
        conv_layer_num:     the Number of convolutional layers
        nz_num:             the Number of non-zero value in each layers
        conv_diff:          the sparse index of each convolutional layers
        fc_diff:            the sparse index of each full-connect layers
        conv_value_array:   the sparse value of each convolutional layers
        fc_value_array:     the sparse value of each full-connect layers
    '''
    conv_layer_num = 0
    fc_layer_num = 0
    fin = open(path, 'rb') # 剪枝后的稀疏文件

    # 实现卷积层和全连接层的计数
    for name, x in net.named_parameters():
        if name.endswith('mask'):
            continue
        if name.startswith('conv'):
            conv_layer_num += 1
        elif name.startswith('fc'):
            fc_layer_num += 1
    # 1 根据文件构造数组，存的是啥读的就是啥：稀疏文件uint32，共 权重层数 个项目
    nz_num = np.fromfile(fin, dtype=np.uint32, count=conv_layer_num + fc_layer_num)

    # 卷积层的 稀疏索引个数 总和
    conv_diff_num = sum(nz_num[:conv_layer_num])
    # 2 读取稀疏文件uint8，共 conv稀疏索引个数 个项目
    conv_diff = np.fromfile(fin, dtype=np.uint8, count=conv_diff_num)

    # 全连接层的稀疏索引个数总和加一，再向下折半
    fc_merge_num = math.floor((sum(nz_num[conv_layer_num:]) + 1) / 2)
    # 4 读取稀疏文件uint8，共fc_merge_num个项目
    fc_merge_diff = np.fromfile(fin, dtype=np.uint8, count=fc_merge_num)
    
    # 3 读取稀疏文件float32，共conv_diff_num个项目
    conv_value_array = np.fromfile(fin, dtype=np.float32, count=conv_diff_num)
    # 5 读取稀疏文件float32，共全连接层的稀疏索引个数总和个项目
    fc_value_array = np.fromfile(fin, dtype=np.float32, count=sum(nz_num[conv_layer_num:]))

    # print(nz_num)
    # print(conv_diff.size, conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # print(conv_value_array.size, conv_value_array[-10:])
    # print(fc_value_array.size, fc_value_array[-10:])

    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 39316 [ 17 242  34  50 164  44  26   3   6 128]
    # 8537 [ 0.05500366 -0.0518913  -0.05787839  0.04747333 -0.07086759 -0.07142863
    #  -0.06043605 -0.06711546 -0.0698091  -0.06924898]
    # 78631 [ 0.13233908  0.16305041 -0.171971   -0.1353672   0.16033891 -0.19598335
    #  -0.11460102 -0.32042998 -0.12170218  0.14367148]

    # Split 8 bits index to 4 bits index
    fc_diff = []
    max_bits = (2 ** fc_bits) - 1 # 值为15
    for i in range(len(fc_merge_diff)): 
    # 将折半的fc索引循环，个数加倍，原第i位的uint8拆成前四后四位，赋给第i、i+1位
        fc_diff.append(int(fc_merge_diff[i] >> fc_bits))  # first 4 bits  
        # 右移四位，相当于舍掉后四位，a/16；舍完最大值不会超过15
        fc_diff.append(fc_merge_diff[i] & max_bits)  # last 4 bits 
        # 与15做按位与，相当于舍掉前四位；舍完最大值不会超过15

    fc_num_sum = nz_num[conv_layer_num:].sum()
    if fc_num_sum % 2 != 0: # 若全连接层索引个数为奇数，则舍掉最后一个？
        fc_diff = fc_diff[:fc_num_sum] # 78631个
    fc_diff = np.asarray(fc_diff, dtype=np.uint8)
    # print("if_error_more_15", (fc_diff > 15).sum())
    # print("if_error_less_0", (fc_diff < 0).sum())
    # print("fc_diff", (fc_diff).sum())
    # layer_index = fc_diff[0:0 + nz_num[4]]
    # print(sum(layer_index) + len(layer_index))

    # print(nz_num)
    # print(conv_diff.size, conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # print(conv_value_array.size, conv_value_array[-10:])
    # print(fc_value_array.size, fc_value_array[-10:])

    # [  292    17  8213    15 77747    65   818     1]
    # 8537 [3 1 2 0 3 2 0 1 2 4]
    # 78631 [ 4  2 12  1 10  0  3  0  6  8]
    # 8537 [ 0.05500366 -0.0518913  -0.05787839  0.04747333 -0.07086759 -0.07142863
    #  -0.06043605 -0.06711546 -0.0698091  -0.06924898]
    # 78631 [ 0.13233908  0.16305041 -0.171971   -0.1353672   0.16033891 -0.19598335
    #  -0.11460102 -0.32042998 -0.12170218  0.14367148]

    return conv_layer_num, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array
    # 卷积层个数 每一层非零值的个数 每个卷积层的稀疏索引 每个全连接层的稀疏索引 每个卷积层的稀疏值 每个全连接层的稀疏值

# 2/8 稀疏参数初始化：把参数拉成一维，替换为原索引的值，再重整为原来的维度
def sparse_to_init(net, before_path, prune_fc_bits):
    conv_layer_length, nz_num, sparse_conv_diff, sparse_fc_diff, conv_value_array, fc_value_array \
        = load_sparse_model(net, before_path, prune_fc_bits)
    state_dict = net.state_dict()
    conv_layer_index = 0 # 卷积层索引
    fc_layer_index = 0 # 全连接层索引
    for i, (key, value) in enumerate(state_dict.items()): # 有多少层循环多少次，循环完可更新为原索引的值
        shape = value.shape # value的维度大小
        value = value.view(-1) # 把value拉成一维
        value.zero_()

        if i < conv_layer_length: # 卷积层个数4
            layer_diff = sparse_conv_diff[conv_layer_index:conv_layer_index + nz_num[i]] # 存的是卷积层的稀疏索引
            layer_value = conv_value_array[conv_layer_index:conv_layer_index + nz_num[i]] # 存的是卷积层的稀疏值
            conv_layer_index += nz_num[i] # 存的是卷积层每层所含的元素个数
        else:
            layer_diff = sparse_fc_diff[fc_layer_index:fc_layer_index + nz_num[i]] # 存的是全连接层的稀疏索引
            layer_value = fc_value_array[fc_layer_index:fc_layer_index + nz_num[i]] # 存的是全连接层的稀疏索引
            fc_layer_index += nz_num[i] # 存的是全连接层=层每层所含的元素个数

        dense_index = 0 # 原索引的索引
        sparse_index = 0 # diff索引的索引
        # value为原值，本为全零矩阵，现可在原索引存储原值
        while sparse_index < len(layer_diff):
            dense_index += layer_diff[sparse_index] # 把稀疏值索引不断做累加，可以得到真实索引
            tmp = layer_value[sparse_index].item()
            value[dense_index] = tmp # 更新原的索引value值
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)

# 3/8 返回 新的索引(值等于指定值) 与存放m的key_parameter
# new_index_list (8, 16)   key_parameter (4, 16)
def restructure_index(index_list, conv_layer_num, max_conv_bit, max_fc_bit):
    '''load the model which is saved as sparse matrix

    Args:
        index_list:         the index of the codebook
        conv_layer_num:     the Number of convolutional layers
        max_conv_bit:       the bits of each value in convolutional layer
        max_fc_bit:         the bits of each value in full-connect layer

    Returns:
        new_index_list:     Contains the index belonging to each value in codebook
        key_parameter:
    '''
    new_index_list = []

    for i in range(len(index_list)): # 8个
        num = max_conv_bit if i < conv_layer_num else max_fc_bit # 选最大bit，8或4
        tmp_index = []
        for j in range(num): # 设置最大bit循环体，8或4
            tmp_index.append(np.where(np.asarray(index_list[i]) == j)[0].tolist())
            # index_list中的元素值为j的第二维坐标
        new_index_list.append(tmp_index)
        # print(new_index_list) ..399982, 399986]], [[6, 51, 54..

    key_parameter = [] # 值只有0,1,2,3 维度4×【520 25050 400500 5010】
    for j in range(int(len(index_list) / 2)): # 4个
        layer_index = np.concatenate((index_list[2 * j], index_list[2 * j + 1]))
        # 拼接，把第一维中，第i个和第i+1个并成一个，总个数减半为4个
        num = max_conv_bit if j < (conv_layer_num / 2) else max_fc_bit
        # 确定num值，8或4
        empty_parameter = [None] * num # 占位，维度为num,4
        key_parameter.append(empty_parameter)
        # print(len(layer_index)) 520 25050 400500 5010
        for m in range(len(layer_index)): # 循环4次
            # print(m, layer_index[m]) 102252 -1 -- 102253 0 -- 102254 12
            if layer_index[m] != -1 and key_parameter[j][layer_index[m]] is None:
                key_parameter[j][layer_index[m]] = m
    return new_index_list, key_parameter
    # new_index_list (8, 16)   key_parameter (4, 16)

# 4/8 codebook初始化：返回new_index_list, key_parameter
# new_index_list (8, 16)   key_parameter (4, 16)
def codebook_to_init(net, conv_layer_length, nz_num, sparse_conv_diff, sparse_fc_diff, codebook, 
                    max_conv_bit, max_fc_bit):
    state_dict = net.state_dict()
    index_list = []
    conv_layer_index = 0
    fc_layer_index = 0
    for i, (key, value) in enumerate(state_dict.items()):
        shape = value.shape # 得到value的初始形状
        value = value.view(-1) # value打平
        value.zero_()

        index = np.empty_like(value, dtype=np.int16) # 与value维度相同
        index[:] = -1 # index与打平后的value维度相同，值为-1
         # value的元素值置零
        if i < conv_layer_length: # 4，提取出卷积层diff索引，并纪录卷积层的参数总个数
            layer_diff = sparse_conv_diff[conv_layer_index:conv_layer_index + nz_num[i]]
            conv_layer_index += nz_num[i]
        else: # 4，提取出全连接层diff索引，并纪录全连接层的参数总个数
            layer_diff = sparse_fc_diff[fc_layer_index:fc_layer_index + nz_num[i]]
            fc_layer_index += nz_num[i]

        dense_index = 0 # 原索引的索引
        sparse_index = 0 # diff索引的索引
        half_index = int(i / 2) # 只可能是0,1,2,3，表示四个层
        codebook_index_array = codebook.codebook_index[half_index] # [320  15475 247429   3095]
        while sparse_index < len(layer_diff): # 该层的diff参数总个数
            dense_index += layer_diff[sparse_index]
            tmp = codebook.codebook_value[half_index][codebook_index_array[sparse_index]].item()
            # codebook_value: (16, 1)；[]确定层[]具体diff索引  找出对应的值
            value[dense_index] = tmp # 将原索引位置换上对应值
            index[dense_index] = int(codebook_index_array[sparse_index]) # 将原索引位置换上codebook索引
            sparse_index += 1
            dense_index += 1
        value.reshape(shape)
        index.reshape(shape)
        index_list.append(index)

    new_index_list, key_parameter = restructure_index(index_list, conv_layer_length, max_conv_bit,
                                                                  max_fc_bit)
    return new_index_list, key_parameter

# 5/8 更新weight和bias的梯度
def cluster_grad(net, index_list, max_conv_bit, max_fc_bit, conv_layer_length):
    params = list(net.parameters())
    # print('================')
    for i in range(0, len(params), 2): # i遍历层，对于网络层数，步长为2，分别是weight与bias
        param = params[i]
        grad_shape = param.grad.shape
        grad = param.grad
        grad = grad.view(-1)
        index = index_list[i]

        bias = params[i + 1]
        bias_grad_shape = bias.grad.shape
        bias_grad = bias.grad
        bias_grad = bias_grad.view(-1)
        bias_index = index_list[i + 1]

        # start = time.clock()
        # Cluster grad using index, use mean of each class of grad to update weight
        # 使用索引聚类梯度，使用每类梯度的平均值更新权重
        cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
        # 选择cluster_bits，8或4，现为4
        for j in range(cluster_bits): # j 遍历 cluster_bits
            sum_grad = grad[index[j]].sum()
            sum_grad += bias_grad[bias_index[j]].sum()
            grad[index[j]] = sum_grad
            bias_grad[bias_index[j]] = sum_grad
        # end = time.clock()
        # print('update weights time:', round(end - start, 5))

        grad = grad.view(grad_shape)
        params[i].grad = grad.clone()

        bias_grad = bias_grad.view(bias_grad_shape)
        params[i + 1].grad = bias_grad.clone()

# 6/8 更新codebook_centroids值，为量化训练后的网络参数？
def update_codebook(net, codebook, conv_layer_length, max_conv_bit, max_fc_bit, key_parameter):
    params = list(net.parameters())
    for i in range(0, len(params), 2): # 遍历层数遍，步长为2
        # start = time.clock()
        param = params[i]
        param = param.view(-1)
        bias_param = params[i + 1]
        bias_param = bias_param.view(-1)
        layer = torch.cat((param, bias_param))
        half_index = int(i / 2) # 对于lenet，为4
        cluster_bits = max_conv_bit if i < conv_layer_length else max_fc_bit
        # 选择cluster_bits，8或4，现为4
        codebook_centroids = codebook.codebook_value[half_index]
        for j in range(cluster_bits): # 遍历bit次
            if key_parameter[half_index][j] is not None:
                tmp = key_parameter[half_index][j]
                codebook_centroids[j] = layer[tmp]

# 7/8 训练更新权重过程
def train_codebook(max_accuracy, nz_num, conv_diff, fc_diff, retrain_codebook_path, key_parameter, use_cuda,
                   max_conv_bit, max_fc_bit, conv_layer_length,
                   codebook, index_list, testloader, net, trainloader, criterion, optimizer,
                   scheduler, epoch=1, top_5=False):
    for epoch in range(epoch):  # loop over the dataset multiple times 15
        train_loss = []
        net.train()
        # i = 0
        # for inputs, labels in tqdm(trainloader):
        
        start = time.clock()
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
            cluster_grad(net, index_list, max_conv_bit, max_fc_bit, conv_layer_length)
            # 更新weight和bias的梯度
            optimizer.step()  # update weight
            train_loss.append(loss.item())
        end = time.clock()
        print('cluster_grad time:', round(end - start, 5))

        mean_train_loss = np.mean(train_loss)
        print("Epoch:", epoch, "Training Loss: %5f" % mean_train_loss)
        accuracy = test(use_cuda, testloader, net, top_5)
        # core =  top_k_score(use_cuda, trainloader, net, top_5)
        # accuracy = score[0]
        scheduler.step()
    update_codebook(net, codebook, conv_layer_length, max_conv_bit, max_fc_bit, key_parameter)
    # 更新codebook_centroids值
    if accuracy[0] > max_accuracy: # 精度满足要求时，保存codebook
       save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, retrain_codebook_path, net)
       max_accuracy = accuracy

# 8/8 
def save_codebook(conv_layer_length, nz_num, conv_diff, fc_diff, codebook, path, net):
    fc_merge_diff = []

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_diff), fc_diff[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 405510 [0 0 0 0 0 0 0 0 0 0]

    length = len(fc_diff)
    fc_diff = list(fc_diff)
    if length % 2 != 0:
        fc_diff.append(0)
    for i in range(math.floor(len(fc_diff) / 2)):
        fc_merge_diff.append((fc_diff[2 * i] << 4) | fc_diff[2 * i + 1])
    nz_num = np.asarray(nz_num, dtype=np.uint32)
    conv_diff = np.asarray(conv_diff, dtype=np.uint8)
    fc_merge_diff = np.asarray(fc_merge_diff, dtype=np.uint8)

    conv_half_len = int(conv_layer_length / 2)
    conv_codebook_index = []
    for m in range(conv_half_len):
        conv_codebook_index.extend(codebook.codebook_index[m])

    fc_codebook_index = []
    for k in range(conv_half_len, len(codebook.codebook_index)):
        fc_codebook_index.extend(codebook.codebook_index[k])

    codebook_value = []
    for j in range(len(codebook.codebook_value)):
        codebook_value.extend(codebook.codebook_value[j])

    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index), fc_codebook_index[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # 5669 [2, 228, 211, 229, 76, 152, 23, 116, 111, 25]
    # 405510 [10, 11, 5, 6, 9, 7, 5, 7, 12, 5]
    # 544 [-0.11808116, -0.06328904, 0.1446653, 0.051914066, -0.03960273, -0.017428499, -0.017428499, 0.0050489083, 0.22879101, 0.051914066]

    length = len(fc_codebook_index)
    if length % 2 != 0:
        fc_codebook_index.append(0)

    fc_codebook_index = np.asarray(fc_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = []
    for i in range(math.floor((len(fc_codebook_index)) / 2)):
        fc_codebook_index_merge.append(
            (fc_codebook_index[2 * i] << 4) | fc_codebook_index[2 * i + 1])

    conv_codebook_index = np.asarray(conv_codebook_index, dtype=np.uint8)
    fc_codebook_index_merge = np.asarray(fc_codebook_index_merge, dtype=np.uint8)
    codebook_value = np.asarray(codebook_value, dtype=np.float32)

    # print(any(np.isnan(codebook_value)))

    # print(nz_num)
    # print(len(conv_diff), conv_diff[-10:])
    # print(len(fc_merge_diff), fc_merge_diff[-10:])
    # print(len(conv_codebook_index), conv_codebook_index[-10:])
    # print(len(fc_codebook_index_merge), fc_codebook_index_merge[-10:])
    # print(len(codebook_value), codebook_value[-10:])
    # [   304     11   5353      1 400000    500   5000     10]
    # 5669 [ 0  2  0  1  1  1  0  9  8 44]
    # 202755 [0 0 0 0 0 0 0 0 0 0]
    # 5669 [  2 228 211 229  76 152  23 116 111  25]
    # 202755 [200  66  71 152 140 171  86 151  87 197]
    # 544 [-0.11808116 -0.06328904  0.1446653   0.05191407 -0.03960273 -0.0174285
    #  -0.0174285   0.00504891  0.22879101  0.05191407]

    # Set to the same dtype uint8 to save
    nz_num.dtype = np.uint8
    codebook_value.dtype = np.uint8

    sparse_obj = np.concatenate((nz_num, conv_diff, fc_merge_diff, conv_codebook_index,
                                 fc_codebook_index_merge, codebook_value))
    sparse_obj.tofile(path)
