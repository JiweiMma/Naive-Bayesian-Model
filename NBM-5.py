import re

#接收一个大字符串并将其解析为字符串列表
#将字符串转换为字符列表
def textParse(bigString):
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\W+', bigString)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


#将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
#dataSet整理的样本数据集
def createVocabList(dataSet):
    # 创建一个空的不重复列表
    # vocabSet返回不重复的词条列表，也就是词汇表
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

if __name__ == '__main__':
    docList = []; classList = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        # 标记非垃圾邮件，0表示非垃圾文件
        classList.append(0)
    #创建词汇表，不重复
    vocabList = createVocabList(docList)
    print(vocabList)