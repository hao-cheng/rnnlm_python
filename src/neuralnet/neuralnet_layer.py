#!/usr/bin/env python

import numpy as np
DTYPE = np.float32

def compute_range_idx(idx, start_idx, size, capacity):
    cur_idx = start_idx + idx
    if cur_idx >= capacity:
        cur_idx -= capacity
    start = cur_idx * size
    end = (cur_idx+ 1) * size
    return (start, end)

class NeuralNetLayer():
    def __init__(self, is_input=False, is_output=False):
        self.size = 1
        self.capacity= 1
        self.activations = None
        self.grad = None
        self.errors = None
        self.activated = False
        self.errored = False
        self.is_input = is_input
        self.is_output = is_output
        self.cur_first = 0

    def set_size(self, val):
        self.size = val

    def set_capacity(self, val):
        self.capacity= val

    def activation(self, idx=0):
        assert(idx < self.capacity)
        start, end = compute_range_idx(idx, self.cur_first, \
                self.size, self.capacity)
        return self.activations[start:end]

    def error(self, idx=0):
        assert(idx < self.capacity)
        start, end = compute_range_idx(idx, self.cur_first, \
                self.size, self.capacity)

        return self.errors[start:end]

    def AllocateLayers(self):
        assert(self.size > 0)
        assert(self.capacity > 0)
        self.activations = np.zeros([self.capacity * self.size, 1], dtype=DTYPE)
        self.errors = np.zeros([self.capacity * self.size, 1], dtype=DTYPE)
        self.grad = np.zeros([self.size, 1], dtype=DTYPE)

    def ResetLayer(self):
        self.activations[:] = 0
        self.errors[:] = 0
        self.grad[:] = 0

    def Rotate(self):
        self.cur_first += 1
        if self.cur_first == self.capacity:
            self.cur_first = 0

    def ResetActivations(self, idx=0):
        self.activated = False
        start, end = compute_range_idx(idx, self.cur_first, \
                self.size, self.capacity)
        self.activations[start:end] = 0.0

    def ResetErrors(self, idx=0):
        self.errored = False
        start, end = compute_range_idx(idx, self.cur_first, \
                self.size, self.capacity)
        self.errors[start:end] = 0.0
        self.grad[:] = 0.0

    def SigmoidActivation(self, ac_val):
        assert(not self.activated)
        self.activated = True
        #ac_val[:] = (np.tanh(0.5 * ac_val) + 1) * 0.5
        ac_val *= 0.5
        ac_val[:] = np.tanh(ac_val) 
        ac_val += 1
        ac_val *= 0.5
        

    def SigmoidGradient(self, err_val, ac_val):
        assert(self.activated)
        assert(not self.is_input)
        self.errored = True
        #self.grad = ac_val * (1.0 - ac_val)
        self.grad[:] = 1.0
        self.grad -= ac_val
        self.grad *= ac_val
        err_val *= self.grad

    def TanhActivation(self, ac_val):
        assert(not self.activated)
        self.activated = True
        ac_val[:] = np.tanh(ac_val)

    def TanhGradient(self, err_val, ac_val):
        assert(self.activated)
        assert(not self.is_input)
        self.errored = True
        #self.grad = (1.0 + ac_val) * (1.0 - ac_val)
        self.grad[:] = -ac_val
        self.grad *= ac_val
        self.grad += 1
        err_val *= self.grad

    def SoftmaxActivation(self, ac_val):
        assert(not self.activated)
        self.activated = True
        # numerical stability
        ac_val -= np.max(ac_val)
        ac_val[:] = np.exp(ac_val)
        ac_val /= np.sum(ac_val)

    def SoftmaxGradient(self, err_val, ac_val, idx):
        assert(self.activated)
        assert(not self.is_input)
        self.errored = True
        self.grad = -ac_val
        self.grad[idx] += 1.0
        err_val[:] = self.grad
