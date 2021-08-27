from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W)
      loss +=  -scores[y[i]] + np.log(np.sum(np.exp(scores)))
      for j in range(num_class):
        log_out = np.exp(scores[j])/np.sum(np.exp(scores))
        if j == y[i]:
          dW[:,j] += (-1 + log_out)*X[i]
        else:
          dW[:,j] += log_out * X[i]
    
    loss /= num_train
    dW /= num_train

    # norm
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # (num_train,C)
    scores = np.dot(X,W)
    # 防止溢出
    scores -= np.max(scores,axis=1).reshape(-1,1)
    softmax_out = np.exp(scores) / np.sum(np.exp(scores),axis=1).reshape(-1,1)
    loss = - np.sum(np.log(softmax_out[range(num_train),list(y)]))
    loss = loss / num_train
    loss += reg * np.sum(W*W)

    dS = softmax_out.copy()
    dS[range(num_train),list(y)] -= 1
    dW = np.dot(X.T,dS)
    dW = dW / num_train + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
