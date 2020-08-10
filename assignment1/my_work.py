from assignment1.cs231n.data_utils import load_CIFAR10
import numpy as np
from collections import Counter
from tqdm import trange

class KNearestNeighbor(object):
    def __init__(self, h=2):
        self.X_train = None
        self.y_train = None
        self.h = h

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=5):
        X = X.reshape(1, -1)
        X = np.repeat(X, self.X_train.shape[0], axis=0)
        d = self.distance(X, self.X_train)
        neighbor_k = np.argsort(d)[:k]
        neighbor_labels = self.y_train[neighbor_k]
        most_label = Counter(neighbor_labels).most_common(1)[0][0]
        return most_label

    def distance(self, X1, X2):
        d = np.sum(np.abs((X1 - X2) ** self.h), axis=1) ** (1 / self.h)
        return d

    def predict_labels(self, X_test, k=5):
        predict_res = [self.predict(X_test[i], k) for i in trange(X_test.shape[0])]
        predict_res = np.array(predict_res)
        return predict_res


def evaluate(y_test, y_predict):
    correct_nums = np.sum(y_test == y_predict)
    accuracy = correct_nums / y_test.shape[0]
    print("The model accuracy is {:.2f}".format(accuracy))
    return accuracy


def image_process(X_train, y_train, X_test, y_test, num_training, num_test):
    mask_training = list(range(num_training))
    X_train = X_train[mask_training]
    y_train = y_train[mask_training]

    mask_test = list(range(num_test))
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

def validation_image_build(X, Y, valid_prob):
    train_num = int(len(X) * (1 - valid_prob))
    mask_training = list(range(train_num))
    X_train = X[mask_training]
    y_train = Y[mask_training]
    mask_val = list(range(train_num, len(X)))
    X_val = X[mask_val]
    y_val = Y[mask_val]

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    return X_train, y_train, X_val, y_val



if __name__ == '__main__':
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    val_prob = 0.1
    num_training = 5000
    num_test = 500
    k = 5
    h = 1
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    X_train, y_train, X_test, y_test = image_process(X_train, y_train, X_test, y_test, num_training, num_test)
    X_train, y_train, X_val, y_val = validation_image_build(X_train, y_train, val_prob)

    for k in [1, 3, 5, 10, 20, 50, 100]:
        print('k : {}'.format(k))
        classifier = KNearestNeighbor(h)
        classifier.train(X_train, y_train)
        y_predict = classifier.predict_labels(X_val, k)
        evaluate(y_val, y_predict)


