import numpy as np
from random import shuffle

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
  nums_train = 1.0*len(y)
  for i in range(int(nums_train)):
    scores = X[i,:].dot(W)
    # deal with numeric-unstable
    scores -= np.max(scores)
    p = np.exp(scores)/sum(np.exp(scores))
    loss += -np.log(p[y[i]])

    dW += X[i,:].reshape(-1,1).dot(p.reshape(1,-1))
    dW[:,y[i]] -=  X[i,:]

  loss /= nums_train
  loss += 0.5*reg*np.sum(W*W)
  dW /= nums_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1,1)
  scores = np.exp(scores)
  loss = sum(-np.log(scores[range(len(y)),y] / np.sum(scores, axis=1)))
  loss = loss/float(len(y)) + 0.5*reg*np.sum(W*W)

  P = scores / np.sum(scores, axis=1).reshape(-1,1)
  P[range(len(y)), y] -= 1
  dW = X.T.dot(P) 
  dW = dW/float(len(y)) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

