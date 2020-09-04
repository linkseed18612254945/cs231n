from data_utils import *
from collections import Counter
from tqdm import tqdm
import utils
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from torch.nn import functional as F
from torch import nn
from torch import optim
import train_utils

class KNNModel(object):
    def __init__(self, h=2, k=5):
        self.train_x = None
        self.train_y = None
        self.h = h
        self.k = k

    @utils.logging_time_wrapper
    def train(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y if isinstance(train_y, np.ndarray) else np.array(train_y)

    @utils.logging_time_wrapper
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


class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)
        output = self.output_layer(x)
        output = F.log_softmax(output)
        return output

class SVMModel(object):
    def __init__(self):
        self.model = svm.LinearSVC()

    @utils.logging_time_wrapper
    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    @utils.logging_time_wrapper
    def predict(self, test_x):
        return self.model.predict(test_x)


class LRModel(object):
    def __init__(self):
        self.model = SGDClassifier()

    @utils.logging_time_wrapper
    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    @utils.logging_time_wrapper
    def predict(self, test_x):
        return self.model.predict(test_x)

if __name__ == '__main__':
    train, test, train_loader, test_loader, classes = cifar10_datasets(train_sample_num=5000, test_sample_num=1000, flat_pix=False)
    model = TwoLayerNN(3072, 256, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    train_utils.train(train_loader, model, criterion, optimizer, epoch_size=1)
    # report = utils.evaluate_report(test.targets, labels, target_names=classes)
    # print(report)


