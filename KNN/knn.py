# encoding = utf-8
# author : renlei
# Reference: 统计学习方法
# https://baike.baidu.com/item/kd-tree/2302515?fr=aladdin
'''
K = 6
p = 2
accuary = 0.95
cost time = 40.978426456451416 s.
'''
# mnist数据集标签有十种，即0,1,2,...,9.
import numpy as np
import time
def load_mnist(txt_data):
    f = open(txt_data, 'r')
    fr = f.readlines()
    dataArr = []
    for line in fr:
        feats = line.strip().split(',')
        dataArr.append(feats)
    f.close()
    return np.array(dataArr, dtype=np.float)
def train_test_split(data, train_rate):
    Idx = np.arange(len(data))
    np.random.shuffle(Idx) # shuffle in-place
    trainIdx, testIdx = Idx[:round(train_rate*len(data))], Idx[round(train_rate*len(data)):]
    train, test = data[trainIdx], data[testIdx]
    return train, test

# 为了方便记录 kdtree 每个节点在整个数据中的全局索引
# 这里 将 整个训练数据（包括标签）整个传入，这样可以省去
# 测试阶段 特征和标签的对应索取。
class KDNode(object):
    def __init__(self, value, split, lchild, rchild):
        self.val = value
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
class KDTree(object):
    def __init__(self,K, data):
        self.K =K
        self.data = data
        def createNode(curSplit, data_set):

            if len(data_set) == 0:
                return None
            m = len(data_set)
            # # 最后一列是标签
            # n = len(data_set[0]) -1 
            data_set = data_set[np.argsort(data_set[:,curSplit])]
            split_pos = m//2
            midVal = data_set[split_pos]

            return KDNode(midVal, curSplit,
                createNode((curSplit+1)%self.K, data_set[:split_pos]),
                createNode((curSplit+1)%self.K, data_set[split_pos+1:])
            )

        self.root = createNode(0, self.data)
        return    
    def search(self, x, p = 2):
        nearest = []
        for _ in range(self.K):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)
        def recurve(node):
            if node is not None:
                axis = node.split
                # 判断 curX 属于当前 分裂空间的左边或者右边
                dxnode = node.val[axis]-x[axis]
                if (dxnode) > 0: recurve(node.lchild)
                else: recurve(node.rchild)

                dist = np.power(np.sum(np.power(np.abs(x-node.val), p)), 1/p)
                for ix, point in enumerate(self.nearest): 
                    # 如果当前nearest内i处未标记（-1），或者新点与x距离更近
                    if point[0] < 0 or dist < point[0]:
                        self.nearest = np.insert(
                            self.nearest, ix, [dist, node.val], axis = 0
                        )
                        self.nearest = self.nearest[:-1]
                        break
                # self.nearest 是从小到大排序的
                # 通过统计-1的个数知道从右到左-1的位数
                # 再往左一位(-n-1)就是nearest 集合里距离最大值的位置
                n = list(self.nearest[:,0]).count(-1)
                # 切分轴的距离(dxnode)比nearest中最大的小则存在相交情形存在相交
                if self.nearest[-n-1,0] > abs(dxnode): 
                    # x[axis]< node.val[axis]时，去右边（左边已经遍历了）
                    if (dxnode) > 0: recurve(node.rchild)
                    else: recurve(node.lchild) 
        recurve(self.root)
        return self.nearest
        

class KNN(KDTree):
    def __init__(self, K, data):
        super(KNN, self).__init__(K, data)
    def train(self):
        return
    def test(self, test, p = 2):
        n = len(test)
        y_pre, y_true = [], []
        for t in test:
            y_true.append(int(t[-1]))
            nearest = self.search(t, p)
            tmpData = nearest[:,1]
            y_tmp = []
            for tmp in tmpData:
                y_tmp.append(tmp[-1])
            y_pre.append(np.argmax(np.bincount(y_tmp)))
        
        errCnt = 0
        for i in range(n):
            if y_pre[i] != y_true[i]: errCnt += 1
        acc = 1-errCnt/n
        print("accuary = {}".format(acc))
        return 
        
if __name__ == "__main__":
    start = time.time()
    datas = load_mnist('_Data/mnist.txt')
    Train, Test= train_test_split(datas, 0.8)
    knn = KNN(K = 1, data = Train)
    knn.train()
    knn.test(Test, p = 2)
    end = time.time()
    print("cost time = {} s.".format(end-start))