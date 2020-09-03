from data_utils import *
from collections import Counter
from tqdm import tqdm
import utils
from sklearn import svm

class KNNModel(object):
    def __init__(self, h=2, k=5):
        self.train_x = None
        self.train_y = None
        self.h = h
        self.k = k

    def train(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y if isinstance(train_y, np.ndarray) else np.array(train_y)

    def predict(self, X):
        dis = self.distance(self.train_x, X)
        neighbor_k = np.argsort(dis, axis=1)[:, :self.k]
        neighbor_labels = self.train_y[neighbor_k]
        labels = [Counter(neighbor_labels[i, :]).most_common(1)[0][0] for i in range(neighbor_labels.shape[0])]
        return np.array(labels)

    def distance(self, X1, X2):
        assert X1.shape[1] == X2.shape[1]
        dis = [np.sum(np.abs(X1 - X2[i]) ** self.h, axis=1) ** (1 / self.h) for i in tqdm(range(X2.shape[0]), desc='Distance :')]
        dis = np.array(dis)
        return dis

class SVMModel(object):
    def __init__(self):
        self.model = svm.SVC()

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)


if __name__ == '__main__':
    train, test, _, _, classes = cifar10_datasets(train_sample_num=10000, test_sample_num=1000, flat_pix=True)
    # model = KNNModel()
    model = SVMModel()
    model.train(train.data, train.targets)
    labels = model.predict(test.data)
    report = utils.evaluate_report(test.targets, labels, target_names=classes)
    print(report)


