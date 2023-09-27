from sklearn.naive_bayes import GaussianNB
import numpy as np
import copy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Basebayes:
    """
    alpha: 拉普拉斯平滑的参数
    fitted: 是否训练过  bool
    data:训练数据集 ndarray（个数x特征数）
    label:训练标签 ndarray（个数，）
    _class:分类的个数（类数，）
    class_list: 记录每个类有多少个数据（类数，）
    log_class_priors: 先验对数概率（类数，）
    score(): 输出混淆矩阵和预测精度
    """

    def __init__(self, alpha=1.0, priors=None):
        # 传入priors则指定先验概率
        if priors:
            assert np.isclose(priors.sum(), 1)
        self.priors = priors
        self.alpha = alpha
        self.fitted = False
        self.data = None
        self.label = None
        self._class = None
        self.class_list = None
        self.log_class_priors = None
        self.num_class = None
        self.num_sample = None
        self.num_feature = None

    def _count_class(self):
        # 假设所有类是按0——n-1映射好的
        self._class = np.unique(self.label)
        self.num_class = len(self._class)

    def _init_class_priors(self):
        # 计算先验概率，同时使用拉普拉斯平滑
        if self.priors:
            self.log_class_priors = np.log(self.priors)
        else:
            for x in self.label:
                self.class_list[x] += 1
            self.log_class_priors = (
                    np.log(self.class_list + self.alpha) - np.log(self.num_sample + self.num_class * self.alpha))

    def score(self, test_data, test_label):
        pred_y = self.predict(test_data)
        confusion_m = np.zeros((self.num_class, self.num_class))
        for true_l in range(self.num_class):
            for pred_l in range(self.num_class):
                confusion_m[true_l, pred_l] = np.sum([pred_y[test_label == true_l] == pred_l])

        print("confusion matrix:\n", confusion_m)
        trues = confusion_m.trace()
        print("accuracy:%f%%" % (100 * trues / len(test_label)))


class Bernollibayes(Basebayes):
    """
    log_feature_prob:类条件对数概率（特征数x类数x2）
    fit(): 对模型进行训练
    predict(): 利用训练好的模型进行预测
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.log_feature_prob = None

    def fit(self, data, label):
        if self.fitted:
            self._class = None
        self._init_coef(data, label)
        self.fitted = True

    def _binarize_data(self, data, label=np.zeros(1), train=True):
        # 更换缺省值，二值化数据
        self.num_sample, self.num_feature = data.shape
        data2 = np.nan_to_num(data, nan=0)
        data2 = (data2 != 0) & (data is not None)
        if label.any():
            assert self.num_sample == len(label)
        if train:
            self.data = data2
            self.label = label
        return data2, label

    def _init_feature_prob(self):
        self.log_feature_prob = np.zeros((self.num_feature, self.num_class, 2))
        counter = np.zeros((self.num_feature, self.num_class))  # 数0的个数

        for index, sample in enumerate(self.data):
            tem_class = self.label[index]
            for feature, value in enumerate(sample):
                counter[feature, tem_class] += 1 - value
        # 计算平滑后每个类别与每个变量的对数概率
        self.log_feature_prob[:, :, 0] = np.log(counter + self.alpha) - np.log(self.class_list + 2 * self.alpha)
        self.log_feature_prob[:, :, 1] = np.log(self.class_list.reshape(-1, self.num_class) - counter + self.alpha) \
            - np.log(self.class_list + 2 * self.alpha)

    def _init_coef(self, data, label):
        self._binarize_data(data, label)
        self._count_class()
        self.log_feature_prob = np.zeros((self.num_sample, self.num_class))
        self._init_class_priors()
        self._init_feature_prob()

    def predict(self, test_data):
        #  训练好模型后预测数据
        assert self.fitted
        num_test = len(test_data)
        test_x, _ = self._binarize_data(test_data, train=False)
        log_test_prob = np.zeros((num_test, self.num_class))

        for index, data in enumerate(test_x):
            for cls in range(self.num_class):
                log_test_prob[index, cls] += self.log_class_priors[cls]
                for feature, value in enumerate(data):
                    log_test_prob[index, cls] += self.log_feature_prob[feature, cls, 1] if value \
                        else self.log_feature_prob[feature, cls, 0]

        pred_y = np.argmax(log_test_prob, axis=1)
        return pred_y


class Multinomialbayes(Basebayes):
    """
    data_dict: 存储特征对应数据个数的列表，列表里有num_feature个数的字典
               字典里存有相应特征的对应数据个数（特征数x类数）
    log_feature_prob: 同上，把个数转化为对数概率
    class_list: 每个类所含的数据个数（类数，）
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.data_dict = None
        self.log_feature_prob = None

    def fit(self, data, label):
        if self.fitted:
            self._class = None
        self._init_coef(data, label)
        self.fitted = True

    def _init_coef(self, data, label):
        self._check_data(data, label)
        self._count_class()
        self.log_feature_prob = np.zeros((self.num_sample, self.num_class))
        self.class_list = np.zeros(self.num_class)
        self._init_class_priors()
        self._init_feature_prob()

    def _check_data(self, data, label=np.zeros(1), train=True):
        self.num_sample, self.num_feature = data.shape
        assert (data >= 0).all()
        if label.any():
            assert self.num_sample == len(label)
        if train:
            self.data = data
            self.label = label
        return data, label

    def _init_feature_prob(self, ):
        # 事先处理字典，让每一个特征下的字典对应相同
        self.data_dict = [[] for _ in range(self.num_feature)]
        self.log_feature_prob = [[] for _ in range(self.num_feature)]
        for col in range(self.num_feature):
            choices = np.unique(self.data[:, col])
            dic = {}
            for choice in choices:
                dic[choice] = 0
            self.data_dict[col] = [copy.deepcopy(dic) for _ in range(self.num_feature)]
            self.log_feature_prob[col] = [copy.deepcopy(dic) for _ in range(self.num_feature)]

        for index, sample in enumerate(self.data):
            tem_class = self.label[index]
            for feature, value in enumerate(sample):
                self.data_dict[feature][tem_class][value] += 1

        # 计算平滑后每个类别与每个变量的对数概率
        for feature in range(self.num_feature):
            feature_choice = max(len(dic) for dic in self.data_dict[feature])
            for cls in range(self.num_class):
                for k, v in self.data_dict[feature][cls].items():
                    self.log_feature_prob[feature][cls][k] = np.log(v + self.alpha) - \
                        np.log(self.class_list[cls] + feature_choice * self.alpha)

    def predict(self, test_data):
        assert self.fitted
        num_test = len(test_data)
        test_x, _ = self._check_data(test_data, train=False)
        log_test_prob = np.zeros((num_test, self.num_class))

        for index, data in enumerate(test_x):
            for cls in range(self.num_class):
                log_test_prob[index, cls] += self.log_class_priors[cls]
                for feature, value in enumerate(data):
                    # 如果遇到了没有记录过的值，说明对任何类都无影响，直接赋0即可
                    log_test_prob[index, cls] += self.log_feature_prob[feature][cls].get(value, 0)
        pred_y = np.argmax(log_test_prob, axis=1)
        return pred_y


class Gaussianbayes(Basebayes):
    """
    epsilon: 计算方差时加上的小数，防止除0错误
    mean: 各特征各类的均值（类数，特征数）
    variance: 各特征各类的方差（类数,特征数）
    _normal: 正态分布函数
    """
    def __init__(self, *args, epsilon=1e-9):
        super().__init__(*args)
        self.epsilon = epsilon
        self.mean = None
        self.varance = None

    def fit(self, data, label):
        if self.fitted:
            self._class = None
        self._init_coef(data, label)
        self.fitted = True

    def _init_coef(self, data, label):
        self._check_data(data, label, train=True)
        self._count_class()
        self.class_list = np.zeros(self.num_class)
        self._init_class_priors()
        self._init_mean_variance()

    def _check_data(self, data, label=np.zeros(1), train=True):
        self.num_sample, self.num_feature = data.shape
        if label.any():
            assert self.num_sample == len(label)
        if train:
            self.data = data
            self.label = label
        return data, label

    def _init_mean_variance(self):
        self.mean = np.zeros((self.num_class, self.num_feature))
        self.variance = np.zeros((self.num_class, self.num_feature))
        for cls in range(self.num_class):
            self.mean[cls, :] = np.mean(self.data[self.label == cls], axis=0)
            self.variance[cls, :] = np.var(self.data[self.label == cls], axis=0)+self.epsilon

    def predict(self, test_data):
        assert self.fitted
        num_test = len(test_data)
        test_x, _ = self._check_data(test_data, train=False)
        log_test_prob = np.zeros((num_test, self.num_class))

        for index, data in enumerate(test_x):
            for cls in range(self.num_class):
                log_test_prob[index, cls] += self.log_class_priors[cls]
                for feature, value in enumerate(data):
                    log_test_prob[index, cls] += \
                        np.log(self._normal(value, self.mean[cls][feature], self.variance[cls][feature]))
        pred_y = np.argmax(log_test_prob, axis=1)
        return pred_y

    def _normal(self, x, m, v):
        # 防止取log后绝对值太大导致-inf
        return max(np.exp(-(x-m)**2/(2*v**2))/np.sqrt(2*np.pi)/v, 1e-9)


if __name__ == "__main__":
    iris = load_iris()
    X,y = iris.data, iris.target
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)
    clf = Gaussianbayes()
    clf2 = GaussianNB()
    clf.fit(x_train,y_train)
    clf.score(X,y)
    clf2.fit(x_train,y_train)
    print(clf2.score(X,y))
