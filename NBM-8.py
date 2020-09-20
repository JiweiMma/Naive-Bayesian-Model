import os
import random
import jieba

#中文文本处理
#folder_path - 文本存放的路径
#test_size - 测试集占比，默认占所有数据集的百分之20
def TextProcessing(folder_path, test_size = 0.2):
    # 查看folder_path下的文件
    folder_list = os.listdir(folder_path)
    # 数据集数据
    data_list = []
    # 数据集类别
    class_list = []

    #遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径
        new_folder_path = os.path.join(folder_path, folder)
        # 存放子文件夹下的txt文件的列表
        files = os.listdir(new_folder_path)

        j = 1
        #遍历每个txt文件
        # 每类txt样本数最多100个
        for file in files:
            if j > 100:
                break
            # 打开txt文件
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:
                raw = f.read()

            # 精简模式，返回一个可迭代的generator
            word_cut = jieba.cut(raw, cut_all = False)
            # generator转换为list
            word_list = list(word_cut)

            # 添加数据集数据
            data_list.append(word_list)
            # 添加数据集类别
            class_list.append(folder)
            j += 1

    # zip压缩合并，将数据与标签对应压缩
    data_class_list = list(zip(data_list, class_list))
    # 将data_class_list乱序
    random.shuffle(data_class_list)
    # 训练集和测试集切分的索引值
    index = int(len(data_class_list) * test_size) + 1
    # 训练集
    train_list = data_class_list[index:]
    # 测试集
    test_list = data_class_list[:index]
    # train_data_list - 训练集列表
    # test_data_list - 测试集列表
    #train_class_list - 训练集标签列表
    #test_class_list - 测试集标签列表
    # 训练集解压缩
    train_data_list, train_class_list = zip(*train_list)
    # 测试集解压缩
    test_data_list, test_class_list = zip(*test_list)

    # all_words_list - 按词频降序排序的训练集列表
    # 统计训练集词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    #根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
    # 解压缩
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    # 转换成列表
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

if __name__ == '__main__':
    #文本预处理
    # 训练集存放地址
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    print(all_words_list)