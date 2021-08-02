# Main File for Learned Index

from __future__ import print_function
import pandas as pd
from Trained_NN import TrainedNN, AbstractNN, ParameterPool, set_data_type
from btree import BTree
from data.create_data import create_data, Distribution
import time, gc, json
import os, sys, getopt
from tqdm import tqdm
import numpy as np
# todo: 这里由于numpy版本, 会打印出来一些warning, 不会影响性能吧最后, 最后弄成numpy=1.16.0也行..
# 为后面186行的保存做准备
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
# ————————————————
# 版权声明：本文为CSDN博主「Zhou_yongzhe」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/Zhou_yongzhe/article/details/87692052

# Setting 
# BLOCK_SIZE = 100
BLOCK_SIZE = 10000  # todo: 这里两个变量的意义
# TOTAL_NUMBER = 300000
TOTAL_NUMBER = 3000000
# todo 这里的这两个值, 有必要跟vreat_data里面的一样吗
# 一个性能差距几乎已经很小的结果↓
# BLOCK_SIZE = 1000
# TOTAL_NUMBER = 300000

# data files
filePath = {
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"
}

# result record path
pathString = {
    Distribution.RANDOM: "Random",
    Distribution.BINOMIAL: "Binomial",
    Distribution.POISSON: "Poisson",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal"
}

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
    Distribution.RANDOM: [1, 4],    
    Distribution.EXPONENTIAL: [55, 10000]
}   

# whether use threshold to stop train for models in stages
useThresholdPool = {
    Distribution.RANDOM: [True, False],    
    Distribution.EXPONENTIAL: [True, False],    
}

# hybrid training structure, 2 stages
def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums,
                    keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
    """
    Args:
        threshold:  从NN替换为BTree的误差限度 todo:可能不太准确
        use_threshold:  是否使用该种替换
        stage_nums:  模型有几个阶级
        core_nums: 核数目, 应该是值得每个阶级有多少个核吧
        train_step_nums: todo:不知道
        batch_size_nums: 应该是在各个阶级训练的时候, 模型的batch_size
        learning_rate_nums: 应该是在各个阶级训练的时候, 模型的学习率
        keep_ratio_nums: todo 不知道
        train_data_x: 用预训练的键
        train_data_y: 用预训练的值
        test_data_x: 用于测试的键
        test_data_y: 用于测试的值
    Returns:
        训练好的模型index
    """
    stage_length = len(stage_nums)  # 阶级数, todo 这里只用了两个阶级, 需要扩展
    col_num = stage_nums[1]  # 第二阶级(初始一阶的话)的模型数量
    # initial
    # tmp_inputs和tmp_labels是三级列表, 每个单位是一个空列表. index是一个二级列表, 每个单位是一个None.
    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]
    # todo: 微小问题, 这里为第一个个阶级的模型开辟的空间有点大, 一个就行了, 开辟了10个

    # 给训练数据和标签赋值
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x  # todo 简单: 好像没有测试test集合?
    for i in range(0, stage_length):  # 两个阶级
        for j in tqdm(range(0, stage_nums[i])):  # 两个阶级各1, 10组数据.
            if len(tmp_labels[i][j]) == 0:
                continue  # todo: 简单: 没有分到数据是什么情况?
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                # first stage, calculate how many models in next stage
                divisor = stage_nums[i + 1] * 1.0 / (TOTAL_NUMBER / BLOCK_SIZE)
                # 数据集的总量除以block的大小, 结果是数据集实际装满了多少个block (TOTAL_NUMBER / BLOCK_SIZE)
                # 模型数除以block数, 结果是一个有效block对应divisor个下层模型.
                # 如果下层模型多, 比如是有效block的三倍数量. 则divisor=3, 也即一个block对应三个模型.
                # 第一层的key本来是根据BLOCK_SIZE确定的, todo 至少应该是这样, 确认下
                for k in tmp_labels[i][j]:
                    # 对key进行缩放.
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:  # 这里设定的是第一层的标签需要缩放, 后面不需要. todo 应该不适配后面几层模型数量不一样多的情况
                labels = tmp_labels[i][j]
                test_labels = test_data_y
            # train model                    
            tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                    learning_rate_nums[i],
                                    keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)            
            tmp_index.train()      
            # get parameters in model (weight matrix and bias matrix)      
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            del tmp_index
            gc.collect()
            if i < stage_length - 1:  # 非最后一个阶级
                # allocate data into training set for models in next stage
                for ind in range(len(tmp_inputs[i][j])):  # 第i个阶级, 第j个模块, 遍历每一个输入的key
                    # 根据本阶级模块输出, 选择下一个阶级所用的模块
                    p = index[i][j].predict(tmp_inputs[i][j][ind])  # 这里的p, 前文的label已经缩放过, 因此不用再缩放 todo 关键还是BLOCK_SIZE TOTAL_SIZE这两个变量的意义弄明白
                    if p > stage_nums[i + 1] - 1:  # 超范围的预测处理方式是使他在范围顶端
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    # 现在处理最后一个模块的事情, 也即用BTree代替部分模块.
    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:  # 模型不存在则跳过, 这种条件编程一定要注意, 否则报错, 甚至莫名其妙的错误
            continue
        mean_abs_err = index[stage_length - 1][i].mean_err
        if mean_abs_err > threshold[stage_length - 1]:
            # replace model with BTree if mean error > threshold
            print("Using BTree")
            index[stage_length - 1][i] = BTree(2)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
    return index

# main function for training idnex
def train_index(threshold, use_threshold, distribution, path):
    # data = pd.read_csv("data/random_t.csv", header=None)
    # data = pd.read_csv("data/exponential_t.csv", header=None)
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)
    # read parameter
    if distribution == Distribution.RANDOM:
        parameter = ParameterPool.RANDOM.value
    elif distribution == Distribution.LOGNORMAL:
        parameter = ParameterPool.LOGNORMAL.value
    elif distribution == Distribution.EXPONENTIAL:
        parameter = ParameterPool.EXPONENTIAL.value
    elif distribution == Distribution.NORMAL:
        parameter = ParameterPool.NORMAL.value
    else:
        return
    stage_set = parameter.stage_set
    # set number of models for second stage (1 model deal with 10000 records)
    # todo: 在调试: 下面这句话是个自适应的操作, 我们把他隐掉, 使用自带的10个模型
    # stage_set[1] = int(round(data.shape[0] / 10000))

    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    # 现在是full train模式, 全部的数据都用作训练
    global TOTAL_NUMBER  # 现在还是1500
    TOTAL_NUMBER = data.shape[0]  # 现在变成了1000, 被覆盖了
    for i in range(data.shape[0]):
        # train_set_x.append(data.ix[i, 0])
        # train_set_y.append(data.ix[i, 1])
        train_set_x.append(data.iloc[i, 0])
        train_set_y.append(data.iloc[i, 1])

    # 在这个模式下, 如果要使用测试集, 也是使用全部的训练集作为测试集
    test_set_x = train_set_x[:]
    test_set_y = train_set_y[:]
    # data = pd.read_csv("data/random_t.csv", header=None)
    # data = pd.read_csv("data/exponential_t.csv", header=None)
    # for i in range(data.shape[0]):
    #     test_set_x.append(data.ix[i, 0])
    #     test_set_y.append(data.ix[i, 1])
    #     test_set_x.append(data.iloc[i, 0])
    #     test_set_y.append(data.iloc[i, 1])

    print("*************start Learned NN************")
    print("Start Train")
    start_time = time.time()
    # train index
    trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, [], [])
    end_time = time.time()
    learn_time = end_time - start_time
    print("训练神经网络所用时间", learn_time)
    print("训练结束, 开始进行测试. 当前试验阶段, 测试集是训练集本身.")
    err = 0
    start_time = time.time()
    # calculate error
    for ind in range(len(test_set_x)):
        # pick model in next stage
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        # predict position
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])
        # 专为调试
        if ind == 1500:
            print("")
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time %f " % search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")
    # write parameter into files
    result_stage1 = {0: {"weights": trained_index[0][0].weights, "bias": trained_index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):  # 不过这里已经是第二个模型了
        if trained_index[1][ind] is None:  # 如果模型空, 直接离开. 比如第一个阶级的模型只有一个, 而不是train_index模型形状这样的10个.
            continue
        if isinstance(trained_index[1][ind], BTree):  # 保存最后的模型, 是B树保存B树模型, 否则保存NN模型
            tmp_result = []
            for ind, node in trained_index[1][ind].nodes.items():
                item = {}
                for ni in node.items:
                    if ni is None:
                        continue
                    item = {"key": ni.k, "value": ni.v}
                tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
                       "numberOfkeys": node.numberOfKeys}
                tmp_result.append(tmp)
            result_stage2[ind] = tmp_result
        else:
            result_stage2[ind] = {"weights": trained_index[1][ind].weights,
                                  "bias": trained_index[1][ind].weights}
    # 最后的模型
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    # with open("model/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json", "wb") as jsonFile:
    with open("model/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json",
                  "w") as jsonFile:
        json.dump(result, jsonFile, cls=NpEncoder)

    # wirte performance into files
    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize(
                          "model/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json")}
    with open("performance/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json",
              "w") as jsonFile:
        json.dump(performance_NN, jsonFile, cls=NpEncoder)

    del trained_index
    gc.collect()
    
    # build BTree index
    print("*************start BTree************")
    bt = BTree(2)
    print("Start Build")
    start_time = time.time()
    bt.build(test_set_x, test_set_y)
    end_time = time.time()
    build_time = end_time - start_time
    print("Build BTree time ", build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre = bt.predict(test_set_x[ind])
        err += abs(pre - test_set_y[ind])
        if err != 0:  # 预测不对的话, 往左右两边查找. todo 重要: BTree还能预测不对吗?
            flag = 1
            pos = pre
            off = 1
            while pos != test_set_y[ind]:
                pos += flag * off
                flag = -flag
                off += 1            
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end BTree************")

    # write BTree into files
    result = []
    for ind, node in bt.nodes.items():
        item = {}
        for ni in node.items:
            if ni is None:
                continue
            item = {"key": ni.k, "value": ni.v}
        tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
               "numberOfkeys": node.numberOfKeys}
        result.append(tmp)

    with open("model/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json",
              "w") as jsonFile:
        json.dump(result, jsonFile, cls=NpEncoder)

    # write performance into files
    performance_BTree = {"type": "BTree", "build time": build_time, "search time": search_time,
                         "average error": mean_error,
                         "store size": os.path.getsize(
                             "model/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json")}
    with open("performance/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json",
              "w") as jsonFile:
        json.dump(performance_BTree, jsonFile, cls=NpEncoder)

    del bt
    gc.collect()


# Main function for sampel training
def sample_train(threshold, use_threshold, distribution, training_percent, path):
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)
    #read parameters
    if distribution == Distribution.RANDOM:
        parameter = ParameterPool.RANDOM.value
    elif distribution == Distribution.LOGNORMAL:
        parameter = ParameterPool.LOGNORMAL.value
    elif distribution == Distribution.EXPONENTIAL:
        parameter = ParameterPool.EXPONENTIAL.value
    elif distribution == Distribution.NORMAL:
        parameter = ParameterPool.NORMAL.value
    else:
        return
    stage_set = parameter.stage_set
    stage_set[1] = int(data.shape[0] * training_percent / 10000)
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]
    interval = int(1 / training_percent)
    # pick data for training according to training percent
    if training_percent != 0.8:
        for i in range(TOTAL_NUMBER):
            test_set_x.append(data.ix[i, 0])
            test_set_y.append(data.ix[i, 1])
            if i % interval == 0:
                train_set_x.append(data.ix[i, 0])
                train_set_y.append(data.ix[i, 1])
    else:
        for i in range(TOTAL_NUMBER):
            test_set_x.append(data.ix[i, 0])
            test_set_y.append(data.ix[i, 1])
            if i % 5 != 0:
                train_set_x.append(data.ix[i, 0])
                train_set_y.append(data.ix[i, 1])

    print("*************start Learned NN************")
    print("Start Train")
    start_time = time.time()
    trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, test_set_x, test_set_y)
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time ", learn_time)
    print("Calculate Error")
    err = 0
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")
    result_stage1 = {0: {"weights": trained_index[0][0].weights, "bias": trained_index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):
        if trained_index[1][ind] is None:
            continue
        if isinstance(trained_index[1][ind], BTree):
            tmp_result = []
            for ind, node in trained_index[1][ind].nodes.items():
                item = {}
                for ni in node.items:
                    if ni is None:
                        continue
                    item = {"key": ni.k, "value": ni.v}
                tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
                       "numberOfkeys": node.numberOfKeys}
                tmp_result.append(tmp)
            result_stage2[ind] = tmp_result
        else:
            result_stage2[ind] = {"weights": trained_index[1][ind].weights,
                                  "bias": trained_index[1][ind].bias}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    with open("model/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json",
              "w") as jsonFile:
        json.dump(result, jsonFile, cls=NpEncoder)

    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize(
                          "model/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json")}
    with open("performance/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json",
              "w") as jsonFile:
        json.dump(performance_NN, jsonFile, cls=NpEncoder)

    del trained_index
    gc.collect()

# help message
def show_help_message(msg):
    help_message = {'command': 'python Learned_BTree.py -t <Type> -d <Distribution> [-p|-n] [Percent]|[Number] [-c] [New data] [-h]',
                    'type': 'Type: sample, full',
                    'distribution': 'Distribution: random, exponential',
                    'percent': 'Percent: 0.1-1.0, default value = 0.5; sample train data size = 300,000',
                    'number': 'Number: 10,000-1,000,000, default value = 300,000',
                    'new data': 'New Data: INTEGER, 0 for no creating new data file, others for creating, default = 1',
                    'fpError': 'Percent cannot be assigned in full train.',
                    'snError': 'Number cannot be assigned in sample train.',
                    'noTypeError': 'Please choose the type first.',
                    'noDistributionError': 'Please choose the distribution first.'}
    help_message_key = ['command', 'type', 'distribution', 'percent', 'number', 'new data']
    if msg == 'all':
        for k in help_message_key:
            print(help_message[k])

    else:
        print(help_message['command'])
        print('Error! ' + help_message[msg])

# command line
def main(argv):
    # print('现在打印argv')
    # print(argv)
    # print('现在打印argv结束了')

    distribution = None
    per = 0.5
    num = 300000
    is_sample = False
    is_type = False
    is_distribution = False
    do_create = True
    try:
        opts, args = getopt.getopt(argv, "hd:t:p:n:c:")
    except getopt.GetoptError:
        show_help_message('command')
        sys.exit(2)
    for opt, arg in opts:
        arg = str(arg).lower()
        print(opt, arg)
        if opt == '-h':
            show_help_message('all')
            return
        elif opt == '-t':
            if arg == "sample":
                is_sample = True
                is_type = True
            elif arg == "full":
                is_sample = False
                is_type = True
            else:
                show_help_message('type')
                return
        elif opt == '-d':
            if not is_type:
                show_help_message('noTypeError')
                return
            if arg == "random":
                distribution = Distribution.RANDOM
                is_distribution = True
            elif arg == "exponential":
                distribution = Distribution.EXPONENTIAL
                is_distribution = True
            else:
                show_help_message('distribution')
                return
        elif opt == '-p':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            per = float(arg)
            if not 0.1 <= per <= 1.0:
                show_help_message('percent')
                return

        elif opt == '-n':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            if is_sample:
                show_help_message('snError')
                return
            num = int(arg)
            if not 10000 <= num <= 1000000:
                show_help_message('number')
                return

        elif opt == '-c':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            do_create = not (int(arg) == 0)

        else:
            print("Unknown parameters, please use -h for instructions.")
            return

    if not is_type:
        show_help_message('noTypeError')
        return
    if not is_distribution:
        show_help_message('noDistributionError')
        return
    if do_create:
        create_data(distribution, num)
    if is_sample:        
        sample_train(thresholdPool[distribution], useThresholdPool[distribution], distribution, per, filePath[distribution])
    else:
        train_index(thresholdPool[distribution], useThresholdPool[distribution], distribution, filePath[distribution])


if __name__ == "__main__":
    # 原作者的代码是命令行方式, 不方便调试, 其实也就是一个列表而已
    my_argv = ['-t', 'full', '-d', 'random', '-n', '100000', '-c', '1']
    main(my_argv)
