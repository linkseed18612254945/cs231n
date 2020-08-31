import numpy as np
from lecture3 import softmax_loss_batch, predict_by_weight
import seaborn
from matplotlib import pyplot as plt
from experiments import cifar10_datasets
import tqdm

def random_optimize(X_train, Y_train):
    best_loss = np.inf
    best_w = None
    all_losses = []
    for _ in tqdm.tqdm(range(1000)):
        W = np.random.randn(10, X_train.shape[1]) * 0.0001
        loss = softmax_loss_batch(X_train, Y_train, W, batch_first=True)
        all_losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_w = W
    return best_w, all_losses

def random_step_optimize(X_train, Y_train):
    best_loss = np.inf
    W = np.random.randn(10, X_train.shape[1]) * 0.0001
    step_size = 0.0001
    for _ in tqdm.tqdm(range(1000)):
        W_try = W + step_size * np.random.randn(10, X_train.shape[1])
        loss = softmax_loss_batch(X_train, Y_train, W_try, batch_first=True)
        if loss < best_loss:
            best_loss = loss
            W = W_try
    return W, best_loss


def eval_numerical_gradient(f, x):
  """
  一个f在x处的数值梯度法的简单实现
  - f是只有一个参数的函数
  - x是计算梯度的点
  """

  fx = f(x) # 在原点计算函数值
  grad = np.zeros(x.shape)
  h = 0.00001

  # 对x中所有的索引进行迭代
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # 计算x+h处的函数值
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # 增加h
    fxh = f(x) # 计算f(x + h)
    x[ix] = old_value # 存到前一个值中 (非常重要)

    # 计算偏导数
    grad[ix] = (fxh - fx) / h # 坡度
    it.iternext() # 到下个维度

  return grad

def CIFAR10_loss_fun(W):
  return softmax_loss_batch(X, Y, W, batch_first=True)


if __name__ == '__main__':
    train_data, _ = cifar10_datasets()
    X = train_data.data.reshape(train_data.data.shape[0], -1)
    Y = train_data.targets
    W = np.random.rand(10, 3072) * 0.001
    eval_numerical_gradient(CIFAR10_loss_fun, W)
    # # best_w = np.random.randn(10, X.shape[1]) * 0.0001
    # best_w, loss = random_step_optimize(X, Y)
    # acc = predict_by_weight(X, Y, best_w, batch_first=True)
    # print(acc)

