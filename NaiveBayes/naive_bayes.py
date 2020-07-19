# encoding = utf-8
# author : renlei
# Reference: 统计学习方法
'''
Test accuary = 0.8975
cost time = 1.2083406448364258
'''
import numpy as np
import time
# mnist 数据集的特征 在区间[0,1]之间
# 这里为了便于实现贝叶斯估计，对特征进行二值化
# 设置阈值为 0.5 ，大于0.5是1，小于0.5是0
# 其中每个样本 (x, y) 中特征都是784列
def load_mnist(txt_data):
    f = open(txt_data, 'r')
    fr = f.readlines()
    dataArr = []
    for line in fr:
        feats = line.strip().split(',')
        tmpList = []
        for feat in feats[:-1]:
            if float(feat) >= 0.5:
                tmpList.append(1)
            else:
                tmpList.append(0)
        tmpList.append(int(feats[-1]))
        dataArr.append(tmpList)
    f.close()
    return np.array(dataArr)

def train_test_split(data, train_rate):
    Idx = np.arange(len(data))
    np.random.shuffle(Idx) # shuffle in-place
    trainIdx, testIdx = Idx[:round(train_rate*len(data))], Idx[round(train_rate*len(data)):]
    train, test = data[trainIdx], data[testIdx]
    return train, test

class Bayes(object):
    def __init__(self):
        return 
    
    def getProbaility(self, x_train, y_train):
        sampleNum, self.featNum= x_train.shape
        label, uniqueFeat = np.unique(y_train), np.unique(x_train)
        self.labelNum, self.featNum = len(label), len(uniqueFeat)
        # 根据式 4.11计算p_y
        self.p_y = [0]*self.labelNum
        for y in y_train:
            self.p_y[y] += 1
        for ck in range(self.labelNum):
            self.p_y[ck] = (self.p_y[ck]+1)/(sampleNum+self.labelNum)

        #计算条件概率 Px_y=P（X=x|Y = y）
        #计算条件概率分成了两个步骤，下方第一个大for循环用于累加，
        #参考书中“4.2.3 贝叶斯估计 式4.10”，下方第一个大for循环内部是
        #用于计算式4.10的分子，至于分子的+1以及分母的计算在下方第二个大For内
        #初始化为全0矩阵，用于存放所有情况下的条件概率
        self.p_xy = np.zeros((self.labelNum, self.featNum, self.featNum))
        for i in range(sampleNum):
            curX = x_train[i]
            curY = y_train[i]
            for j in range(self.featNum):
                self.p_xy[curY, j, curX[j]] += 1
        
        #第二个大for，计算式4.10的分母，以及分子和分母之间的除法
        #循环每一个标记（共10个）
        for ck in range(self.labelNum):
            #循环每一个标记对应的每一个特征
            for j in range(self.featNum):
                #获取y=ck，第j个特诊为0的个数
                Px_y0 = self.p_xy[ck][j][0]
                #获取y=ck，第j个特诊为1的个数
                Px_y1 = self.p_xy[ck][j][1]
                #对式4.10的分子和分母进行相除，再除之前依据贝叶斯估计，
                #分母需要加上uniqueself.featNum（2）
                #分别计算对于y= ck，x第j个特征为0和1的条件概率分布
                self.p_xy[ck][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
                self.p_xy[ck][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

        return np.log(self.p_y), self.p_xy
    #转换为log对数形式
    #最后求后验概率估计的时候，形式是各项的相乘（“4.1 朴素贝叶斯法的学习” 式4.7），
    # 这里存在两个问题：
    # 1.某一项为0时，结果为0.这个问题通过分子和分母加上一个相应的数可以排除，前面已经做好了处理。
    # 2.如果特诊特别多（例如在这里，需要连乘的项目有784个特征
    #加一个先验概率分布一共795项相乘，所有数都是0-1之间，结果一定是一个很小的接近0的数。）
    # 理论上可以通过结果的大小值判断， 但在程序运行中很可能会向下溢出无法比较，因为值太小了。
    # 所以人为把值进行log处理。log在定义域内是一个递增函数，也就是说log（x）中，
    #x越大，log也就越大，单调性和原数据保持一致。
    # 所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。

    def train(self, x_train, y_train):
        self.p_y, self.p_xy = self.getProbaility(x_train, y_train)
        return
    def getMaxPro(self, x_test):
        p = [0]*self.labelNum
        for ck in range(self.labelNum):
            p[ck] += self.p_y[ck]
            for j in range(self.featNum):
                p[ck] += self.p_xy[ck][j][x_test[j]]
        # 找到该概率最大值对应的索引
        return np.argmax(p)
    def test(self, x_test, y_test):
        errCnt = 0
        for ix, x_te in enumerate(x_test):
            if y_test[ix] == self.getMaxPro(x_te): errCnt += 1
        acc = 1-errCnt/len(x_test)
        print("Test accuary = {}".format(acc))
        return

if __name__ == "__main__":
    start = time.time()
    data = load_mnist('./_Data/mnist.txt')
    Train, Test= train_test_split(data, 0.8)
    bay = Bayes()
    bay.train(Train[:,:-1], Train[:,-1])
    bay.test(Test[:,:-1],Test[:,-1])
    end = time.time()
    print("cost time = {}".format(end-start))