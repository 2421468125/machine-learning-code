import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KdTreeNode:
    """
    Kd树的结点类，重写了[]方法
    """
    def __init__(self, value, cls, layer):
        self.value = value
        self.class_ = cls
        self.layer = layer
        self.left = None
        self.right = None
        self.father = None

    def __getitem__(self, item):
        return self.value[item]


# 计算距离的函数，这里用欧几里得距离
def distance(a, b):
    return (np.sum((a - b)**2))**.5


class KdTree:
    """
    Kd树
    _bulid_tree：通过递归进行建树，循环对特征进行二分直到仅剩1个结点
    prior_print：先序遍历
    search：搜索前k个离target最近的点
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.num_sample = data.shape[0]
        self.num_feature = data.shape[1]
        self.root = None
        self._bulid_tree()

    def _bulid_tree(self, layer=0, data=None, father=None):
        if layer == 0:
            data = np.array([(self.data[i], self.label[i]) for i in range(self.num_sample)], dtype=tuple)
        elif not data:
            return

        mid = len(data) // 2
        feature = layer % self.num_feature
        sorted_data = sorted(data, key=lambda x: x[0][feature])
        node = KdTreeNode(sorted_data[mid][0], sorted_data[mid][1], layer)
        if layer == 0:
            self.root = node
        node.father = father
        node.left = self._bulid_tree(layer=layer+1, data=sorted_data[:mid], father=node)
        node.right = self._bulid_tree(layer=layer+1, data=sorted_data[mid+1:], father=node)
        return node

    def prior_print(self):
        self._prior_print(self.root)

    def _prior_print(self, node):
        print(node.value)
        if node.left:
            self._prior_print(node.left)
        if node.right:
            self._prior_print(node.right)

    def search(self, k, target):
        ngb = [None for _ in range(k)]
        self._search(self.root, 0xffffff, ngb, target)
        return ngb

    def _search(self, node, min_d, ngb, target, other=False):
        if not node:
            return min_d
        if distance(node.value, target) < min_d:
            ngb.insert(0, node)
            ngb.pop(-1)
            if ngb[-1]:
                # 注意如果这里改用sorted时需要对ngb重写赋值，会改变ngb的地址，导致原ngb与函数内不一样
                ngb.sort(key=lambda x: distance(x.value, target))
                min_d = distance(ngb[-1].value, target)

        feature = node.layer % self.num_feature
        last_feature = feature-1 if feature else self.num_feature-1
        # 进入左子树的条件是目标在左侧且左边非空（防止漏掉右结点），或右边空（防止漏掉左结点）
        if target[feature] < node[feature] and node.left or not node.right:
            min_d = self._search(node.left, min_d, ngb, target)
        else:
            min_d = self._search(node.right, min_d, ngb, target)

        # 是否在兄弟结点为根的子树内
        if node.father and abs(target[last_feature] - node.father[last_feature]) < min_d and not other:
            if node.father.left == node:
                min_d = self._search(node.father.right, min_d, ngb, target, other=True)
            else:
                min_d = self._search(node.father.left, min_d, ngb, target, other=True)

        return min_d

    # 传统方法找前k个邻居，这里写仅供参考
    def find(self, k, target):
        data = np.array([(self.data[i], self.label[i]) for i in range(self.num_sample)], dtype=tuple)
        data = sorted(data, key=lambda x: distance(x[0], target))
        for i in range(k):
            print(data[i][1], data[i][0], distance(data[i][0], target))


class KNN:
    """
    k:超参数，k个邻居
    weight：可选参数，”distance“表示和距离成反比, None表示平均
    epsilon：防止距离为0产生error而加上的一个小数
    num_sample：样本个数
    num_feature：特征个数/维度
    numclass_：类别个数
    """
    def __init__(self, k, weight=None, epsilon=0.01):
        self.k = k
        self.epsilon = epsilon
        self.data = None
        self.label = None
        self.weight = weight
        self.KdTree = None
        self.fitted = False
        self.num_sample = None
        self.num_feature = None
        self.numclass_ = None
        self.class_ = None
        assert weight in (None, "distance")

    def fit(self, train_data, train_label):
        self._check_data(train_data, train_label)
        self._count_class()
        self.KdTree = KdTree(self.data, self.label)

    def predict(self, test_data):
        test_x, _ = self._check_data(test_data, train=False)
        num_test = len(test_x)
        preds = np.zeros(num_test)
        for index, sample in enumerate(test_x):
            preds[index] = self._weighted_aver(self.KdTree.search(self.k, sample), sample)
        return preds

    def score(self, test_data, test_label):
        pred_y = self.predict(test_data)
        confusion_m = np.zeros((self.num_class, self.num_class))
        for true_l in range(self.num_class):
            for pred_l in range(self.num_class):
                confusion_m[true_l, pred_l] = np.sum([pred_y[test_label == true_l] == pred_l])

        print("confusion matrix:\n", confusion_m)
        trues = confusion_m.trace()
        print("accuracy:%f%%" % (100 * trues / len(test_label)))

    def _check_data(self, data, label=np.zeros(1), train=True):
        self.num_sample, self.num_feature = data.shape
        if label.any():
            assert self.num_sample == len(label)
        if train:
            self.data = data
            self.label = label
        else:
            assert data.shape[1] == self.data.shape[1]
        return data, label

    def _count_class(self):
        # 假设所有类是按0——n-1映射好的
        self.class_ = np.unique(self.label)
        self.num_class = len(self.class_)

    def _weighted_aver(self, ngb, target):
        dic = {}
        cls = min_value = 0
        for node in ngb:
            dic[node.class_] = 0
            if not self.weight:
                dic[node.class_] += 1
            elif self.weight == "distance":
                dic[node.class_] += 1/(distance(node.value, target) + self.epsilon)
            if dic[node.class_] > min_value:
                cls, min_value = node.class_, dic[node.class_]
        return cls


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=0)
    clf = KNN(5, weight="distance")
    clf.fit(train_x, train_y)
    clf.score(test_x, test_y)
    clf2 = KNeighborsClassifier(5)
    clf2.fit(train_x, train_y)
    print(clf2.score(X, y))
