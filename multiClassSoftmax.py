import numpy as np

def checkSize(w, X, y):
    # w: 785 by 10 matrix
    # X: N by 785 matrix
    # y: N by 1 matrix
    print(y.dtype)
    assert y.dtype == 'int'
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == w.shape[0]
    assert len(y.shape) == 2
    assert y.shape[1] == 1

def loss(w, X, y):
    """
    Optional
    Useful to run gradient checking
    Utilize softmax function below
    """
    checkSize(w, X, y)

    # TODO

def grad(w, X, y):
    """
    Return gradient of multiclass softmax
    Utilize softmax function below
    """
    checkSize(w, X, y)
    
    # TODO
    temp = np.matmul(X.transpose(), w)
    grad = np.matmul(X, np.multiply(-y, sigma(np.multiply(-y, temp))))
    return grad


def sigma(input):
    return 1./(1 + np.exp(-input))

def softmax(w, X):
    scores = np.matmul(X, w)
    maxscores = scores.max(axis = 1)
    scores = scores - maxscores[:, np.newaxis]
    exp_scores = np.exp(scores)

    sum_scores = np.sum(exp_scores, axis = 1)
    return exp_scores/sum_scores[:, np.newaxis]

def predict(w, X):
    """
    Prediction
    """

    # TODO
    return np.argmax(np.dot(X,w), axis = 1)

def accuracy(w, X, y):
    """
    Accuracy of the model
    """
    
    # TODO
    pred = np.argmax(np.dot(X.T, w), axis = 1)
    result = 0 
    return float(np.sum(np.abs(y - pred) < 0.00001))/len(pred)
