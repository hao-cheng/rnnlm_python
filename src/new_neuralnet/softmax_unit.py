#!/usr/bin/env python

import numpy as np
DTYPE = np.double

class SoftmaxUnit():
    """This class stores the activations of the RNN Unit"""
    def __init__(self, output_size, batch_size, dtype):
        self.p = np.zeros([output_size, batch_size], dtype=dtype)

    """Returns p = softmax(Woh*h)"""
    def forward_function(self, h, Woh):
        self.p = np.dot(Woh, h)
        self.p -= np.max(self.p, 0)
        self.p[:] = np.exp(self.p)
        self.p /= np.sum(self.p, 0)
        return self.p

    """Returns the cross entropy with respect to target"""
    def compute_loss(self, target):
        return np.sum(np.log(self.p[target[0], range(self.p.shape[1])]) * target[1])

    """Returns dE/dh, and dEdWoh, where E = cross entropy"""
    def backward_function(self, target, h, Woh):
        dEdp = -self.p
        dEdp[target[0], range(dEdp.shape[1])] += 1.0
        dEdp *= target[1]
        dWoh = np.dot(dEdp, h.T)
        dEdh = np.dot(dEdp.T, Woh)
        return dEdh.T, dWoh


#Tests for the gradient computation of the single RNNUnit
if __name__ == '__main__':
    output_size = 10
    input_size = 5
    batch_size = 4
    Woh = np.zeros([output_size, input_size], dtype=DTYPE)
    Woh += np.random.uniform(-0.1, 0.1, Woh.shape)
    h = np.zeros([input_size, batch_size], dtype=DTYPE)
    h += np.random.uniform(-0.1, 0.1, [input_size, batch_size])
    target = [None] * 2
    target[0] = [2] * batch_size
    target[1] = [1.0] * batch_size
    target[1][0] = 0.0
    target[1][2] = 0.0
    print target
    
    softmax_unit = SoftmaxUnit(output_size, batch_size, DTYPE)
    
    # Exact gradient computation
    p = softmax_unit.forward_function(h, Woh)
    E = softmax_unit.compute_loss(target)
    dEdh, dWoh = softmax_unit.backward_function(target, h, Woh)
    
    # Numerical gradient computation
    epsilon = 1e-7
    numdWoh = np.zeros([output_size, input_size],dtype=DTYPE)
    for i in range(output_size):
        for j in range(input_size):
            newWoh = np.copy(Woh)
            newWoh[i,j] += epsilon
            
            p = softmax_unit.forward_function(h, newWoh)
            newE = softmax_unit.compute_loss(target)
            numdWoh[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWoh - dWoh)
    assert diff < 1e-3
    print 'Check Passed! Diff is', diff
    exit(0)
