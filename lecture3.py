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

def softmax_loss(x, y, W):
    scores = W.dot(x)
    exp_score = np.exp(scores)
    losses = exp_score / np.sum(exp_score)
    log_losses = -np.log(losses)
    loss = log_losses[y]
    print(loss)
    return loss

def softmax_loss_batch(X, Y, W, batch_first=False):
    if batch_first:
        X = X.transpose()
    scores = W.dot(X)
    exp_scores = np.exp(scores)
    sums = np.sum(exp_scores, axis=0).reshape(1, exp_scores.shape[1])
    losses = exp_scores / sums
    log_losses = -np.log(losses)
    loss = log_losses[Y, np.arange(log_losses.shape[1])]
    batch_loss = np.mean(loss)
    return batch_loss

def predict_by_weight(X, Y, W, batch_first=False):
    if batch_first:
        X = X.transpose()
    scores = W.dot(X)
    Y_predict = np.argmax(scores, axis=0)
    return np.mean(Y_predict == Y)

if __name__ == '__main__':
    y = 2
