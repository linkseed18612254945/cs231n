import numpy as np

def svm_loss_single(x, y, W):
    delta = 1
    scores = W.dot(x)
    correct_score = scores[y]
    loss = 0
    for i in range(x.shape[0]):
        if i == y:
            continue
        loss += max([0, scores[i] - correct_score + delta])
    return loss

def svm_loss_single_matrix(x, y, W):
    delta = 1
    scores = W.dot(x)
    losses = np.maximum(0, scores - scores[y] + delta)
    losses[y] = 0
    loss = np.sum(losses)
    return loss

def batch_loss_single_matrix(X, y, W):
    # TODO
    delta = 1
    scores = W.dot(X.transpose())

    y_score = scores[y, :]

    losses = np.maximum(0, scores - y_score + delta)
    loss = np.sum(losses, axis=1)
    return loss

def softmax(x, y, W):
    scores = W.dot(x)
    exp_score = np.exp(scores)
    losses = exp_score / np.sum(exp_score)
    log_losses = -np.log(losses)
    loss = log_losses[y]
    print(loss)
    return loss

if __name__ == '__main__':
    x = np.array([1, -15, 22, -44, 56])
    W = np.array([[0, 0.2, -0.3], [0.01, 0.7, 0], [-0.05, 0.2, -0.45],[0.1, 0.05, -0.2],[0.05, 0.16, 0.03]])
    W = W.transpose()
    y = 2
    softmax(x, y, W)