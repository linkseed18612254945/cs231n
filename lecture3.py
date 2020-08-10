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
    delta = 1
    scores = W.dot(X.transpose())
    y_score = scores[y, :]
    losses = np.maximum(0, scores - y_score + delta)
    loss = np.sum(losses, axis=1)
    return loss