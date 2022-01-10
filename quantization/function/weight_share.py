import numpy as np
import quantization.function.helper as helper
from sklearn.cluster import KMeans
from quantization.function.netcodebook import NetCodebook

# net为剪枝的网络，before_path为剪枝网络路径，8， 4， 4
# share_weight意义在于将load_sparse_model返回的各个参数，把权重层的稀疏值做kmeans聚类存到codebook替代原值
def share_weight(net, before_path, conv_bits, fc_bits, prune_fc_bits):
    '''quantization

    Args:
        net:            the network object
        before_path:    the path of the pruned model
        conv_bits:      the bits of each value in convolutional layer
        fc_bits:        the bits of each value in full-connect layer        
        prune_fc_bits:  the bits of each index in full-connect layer

    Returns:
        conv_layer_num:     the Number of convolutional layers
        codebook:           he sparse value of each convolutional layers
        nz_num:             the Number of non-zero value in each layers
        conv_diff:          the sparse index of each convolutional layers
        fc_diff:            the sparse index of each full-connect layers
    '''
    conv_layer_num, nz_num, conv_diff, fc_diff, conv_value_array, fc_value_array \
        = helper.load_sparse_model(net, before_path, prune_fc_bits)
    # 卷积层个数 每一层非零值的个数 每个卷积层的稀疏索引 每个全连接层的稀疏索引 每个卷积层的稀疏值 每个全连接层的稀疏值
    conv_index = 0
    fc_index = 0
    # print('size', conv_value_array.shape, 'conv_value_array', conv_value_array)

    # 把NetCodebook实例化，（8， 4）
    codebook = NetCodebook(conv_bits, fc_bits)

    have_bias = True
    stride = 2 if have_bias else 1 # stride=2

    layer_nz_num = nz_num[0::stride] + nz_num[1::stride]
    # print(nz_num)            [308     12  15445     30 247121    308   3089      6]
    # print(layer_nz_num)      [320  15475 247429   3095]
    # print(len(layer_nz_num)) 4
    for i in range(len(layer_nz_num)): # 对于每一层
        layer_type = 'conv' if i < conv_layer_num / stride else 'fc'
        if layer_type == 'fc': # 对于全连接层
            bits = fc_bits # 4
            layer_weight = fc_value_array[fc_index:fc_index + layer_nz_num[i]]
            fc_index += layer_nz_num[i]
        else: # 对于卷积层
            bits = conv_bits # 8
            layer_weight = conv_value_array[conv_index:conv_index + layer_nz_num[i]]
            conv_index += layer_nz_num[i]
        # print('No.', i, 'size', layer_weight.shape, 'layer', layer_weight)

        n_clusters = 2 ** bits # 16或者256
        # Use linear initialization for kmeans
        min_weight = min(layer_weight)
        max_weight = max(layer_weight)
        space = np.linspace(min_weight, max_weight, num=n_clusters, dtype=np.float32)
        
        kmeans = KMeans(n_clusters=n_clusters, init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        # kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, precompute_distances=True,
        #                 algorithm="full")
        kmeans.fit(layer_weight.reshape(-1, 1))
        codebook_index = np.array(kmeans.labels_, dtype=np.uint8)
        # print(new_layer_weight[:5])
        codebook_value = kmeans.cluster_centers_[:n_clusters]
        # print('codebook_index:', codebook_index.shape)
        # print('codebook_value:', codebook_value.shape)
        # codebook_index: (320,) codebook_value: (16, 1)
        # codebook_index: (15475,) codebook_value: (16, 1)
        # codebook_index: (247429,) codebook_value: (16, 1)
        # codebook_index: (3095,) codebook_value: (16, 1)

        codebook.add_layer_codebook(codebook_index.reshape(-1), codebook_value.reshape(-1))
        # codebook存放的是  权重层稀疏值做kmeans聚类后的索引与值

    return conv_layer_num, codebook, nz_num, conv_diff, fc_diff
    # 返回卷积层个数 codebook 每一层非零值的个数 每个卷积层的稀疏索引 每个全连接层的稀疏索引
