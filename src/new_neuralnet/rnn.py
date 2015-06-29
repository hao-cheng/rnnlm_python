#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import cPickle as cp

from rnn_unit import RNNUnit
from softmax_unit import SoftmaxUnit

DTYPE = np.double

class RNN():
    def __init__(self):
        ## optimization parameters
        self.init_range = 0.1
        self.learning_rate = 0.1
        self.verbose = True
        self.batch_size = 1

        ## neuralnet structure params
        self.bptt_unfold_level = 1
        self.input_size = 0;
        self.hidden_size = 0;
        self.output_size = 0;
        
        self.rnn_units = []
        self.softmax_units = []

    '''
    RNN mutators
    '''
    def set_init_range(self, val):
        self.init_range = val

    def set_learning_rate(self, val):
        self.learning_rate = val

    def set_batch_size(self, val):
        self.batch_size = val

    def set_verbose(self, val):
        self.verbose = val

    def set_input_size(self, val):
        self.input_size = val

    def set_hidden_size(self, val):
        self.hidden_size = val

    def set_output_size(self, val):
        self.output_size = val

    def set_bptt_unfold_level(self, val):
        self.bptt_unfold_level = val


    '''
    Model utility functions
    '''
    def InitializeParemters(self):
        ## Randomly initialize the connection weights
        self.Whx += np.random.uniform(-self.init_range, self.init_range, self.Whx.shape)
        self.Whh += np.random.uniform(-self.init_range, self.init_range, self.Whh.shape)
        self.Woh += np.random.uniform(-self.init_range, self.init_range, self.Woh.shape)
        
    def ResetStates(self, idxs=None):
        if idxs is None:
            idxs = range(self.hprev.shape[1])
        self.hprev[:, idxs] = 0

    def AllocateModel(self):
        ## Allocate model parameters
        self.Whx = np.zeros([self.hidden_size, self.input_size], dtype=DTYPE)
        self.Whh = np.zeros([self.hidden_size, self.hidden_size], dtype=DTYPE)
        self.Woh = np.zeros([self.output_size, self.hidden_size], dtype=DTYPE)

        self.last_Whx = np.zeros([self.hidden_size, self.input_size], dtype=DTYPE)
        self.last_Whh = np.zeros([self.hidden_size, self.hidden_size], dtype=DTYPE)
        self.last_Woh = np.zeros([self.output_size, self.hidden_size], dtype=DTYPE)
       
        ## Allocate states
        self.hprev = np.zeros([self.hidden_size, self.batch_size], dtype=DTYPE)
        
        ## Allocate activations and softmax
        for _ in range(self.bptt_unfold_level):
            self.rnn_units.append(RNNUnit(self.hidden_size, self.batch_size, DTYPE))
            self.softmax_units.append(SoftmaxUnit(self.output_size, self.batch_size, DTYPE))

    def ReadModel(self, fname):
        ## Read model from file
        if not os.path.exists(fname):
            sys.stderr.write(\
                'Error: Model file {} does not exist!\n'.format(fname))
            sys.exit(1)

        with open(fname) as fin:
            model = cp.load(fin)
            print '=========Reading Model========\n'
            self.init_range = model['init_range']
            self.input_size = model['input_size']
            self.hidden_size = model['hidden_size']
            self.output_size = model['output_size']
            self.learning_rate = model['learning_rate']
            self.bptt_unfold_level = model['bptt_unfold_level']

            self.Whx = model['Whx']
            self.Whh = model['Whh']
            self.Woh = model['Woh']
            print '=========Reading Done========\n'

    def WriteModel(self, fname):
        ## Write model to file
        model = {}
        model['init_range'] = self.init_range
        model['input_size'] = self.input_size
        model['hidden_size'] = self.hidden_size
        model['output_size'] = self.output_size
        model['learning_rate'] = self.learning_rate
        model['bptt_unfold_level'] = self.bptt_unfold_level

        model['Whx'] = self.Whx
        model['Whh'] = self.Whh
        model['Woh'] = self.Woh

        with open(fname, 'wb') as fout:
            print '=========Writing Model========\n'
            cp.dump(model, fout)
            print '=========Writing Done========\n'

    '''
    Forward propogation
    '''
    def ForwardPropagate(self, input_idxs, target_idxs, ivcount=False, \
            skip_set=[], eval=False):
        if not eval:
            assert len(input_idxs) == self.bptt_unfold_level
        loss = 0
        probs = []
        
        #self.ResetStates() # put this control outside
        iv_count = 0

        for i, (input_idx, target_idx) in enumerate(zip(input_idxs, target_idxs)):
            assert len(input_idx) == self.batch_size
            assert len(target_idx) == 2
            assert len(target_idx[0]) == self.batch_size
            assert len(target_idx[1]) == self.batch_size
            x = np.zeros([self.input_size, self.batch_size], dtype=DTYPE)
            x[input_idx, range(self.batch_size)] = 1.0
            h = self.rnn_units[i].forward_function(x, self.hprev, self.Whx, self.Whh)
            p = self.softmax_units[i].forward_function(h, self.Woh)
            probs += [p]
            if ivcount:
                for ind, word_idx in enumerate(target_idx[0]):
                    if word_idx not in skip_set:
                        iv_count += 1
                        loss += np.log(p[word_idx, ind])
            else:
                loss += self.softmax_units[i].compute_loss(target_idx)
            self.hprev = h 
        return loss, probs, iv_count     


    '''
    Backpropogation through time
    '''
    def BackPropagate(self, input_idxs, target_idxs):
        dWhh = np.zeros(self.Whh.shape)
        dWoh = np.zeros(self.Woh.shape)
        dWhx = np.zeros(self.Whx.shape)
        dEdh = np.zeros([self.hidden_size, self.batch_size])
        for i in range(self.bptt_unfold_level-1, -1, -1):
            target_idx = target_idxs[i]
            input_idx = input_idxs[i]
            #Retrieve activations
            h = self.rnn_units[i].h
            # we might want to keep the sequence and reset it globally
            # hprev = self.rnn_units[i-1].h
            if i > 0:
                hprev = self.rnn_units[i-1].h
            else:
                hprev = np.zeros([self.hidden_size, self.batch_size])
            #Backprop the Softmax
            dEdh_softmax, l_dWoh = self.softmax_units[i].backward_function(target_idx, h, self.Woh)
            
            #Backprop the RNN
            x = np.zeros([self.input_size, self.batch_size], dtype=DTYPE)
            x[input_idx, range(self.batch_size)] = 1.0
            dEdhprev, l_dWhx, l_dWhh = self.rnn_units[i].backward_function(x, hprev, dEdh + dEdh_softmax, 
                                                              self.Whx, self.Whh)
            
            #Update the gradient accumulators
            dEdh = dEdhprev
            dWhh += l_dWhh
            dWoh += l_dWoh
            dWhx += l_dWhx
        return dWhh, dWoh, dWhx

    def UpdateWeight(self, dWhh, dWoh, dWhx):
        dWhh *= self.learning_rate
        dWoh *= self.learning_rate
        dWhx *= self.learning_rate
        self.Whh += dWhh
        self.Woh += dWoh
        self.Whx += dWhx
        
    def RestoreModel(self):
        self.Whh[:] = self.last_Whh
        self.Woh[:] = self.last_Woh
        self.Whx[:] = self.last_Whx
        
    def CacheModel(self):
        self.last_Whh[:] = self.Whh
        self.last_Woh[:] = self.Woh
        self.last_Whx[:] = self.Whx

    def ComputeLogProb(self, input_idx):
        output_layer_activations = self.output_layer.activation(0)
        return np.log(output_layer_activations[input_idx, 0])

    def GetMostProbNext(self):
        output_layer_activations = self.output_layer.activation(0)
        return np.argmax(output_layer_activations)

    def GetMostProbUniqNext(self, cur_seq):
        output_layer_activations = self.output_layer.activation(0)
        sort_indices = [i[0] for i in \
                sorted(enumerate(output_layer_activations), \
                key=lambda x:x[1], \
                reverse=True)]
        for idx in sort_indices:
            if idx not in cur_seq:
                break
        return idx

    def GetMostProbUniqNextFromSet(self, cur_seq, item_set):
        output_layer_activations = self.output_layer.activation(0)
        sort_indices = [i[0] for i in \
                sorted(enumerate(output_layer_activations), \
                key=lambda x:x[1], \
                reverse=True)]
        for idx in sort_indices:
            if item_set[idx] == True and \
                idx not in cur_seq:
                break
        return idx
    
if __name__ == '__main__':
    rnn = RNN()
    rnn.set_batch_size(3)
    rnn.set_bptt_unfold_level(10)
    rnn.set_hidden_size(20)
    rnn.set_input_size(5)
    rnn.set_init_range(0.1)
    rnn.set_learning_rate(0.1)
    rnn.set_output_size(15)
    rnn.AllocateModel()
    rnn.InitializeParemters()
    
    # Fake indices
    input_idxs = []
    target_idxs = []
    for t in range(rnn.bptt_unfold_level):
        input_idx = []
        target_idx = []
        target_mult = []
        for b in range(rnn.batch_size):
            input_ind = np.random.randint(0, rnn.input_size)
            input_idx.append(input_ind)
            target_ind = np.random.randint(0, rnn.output_size)
            target_idx.append(target_ind)
            target_mult.append(1.0)
        input_idxs.append(input_idx)
        target_idxs.append((target_idx, target_mult))
    
    print input_idxs
    print target_idxs
    # Numerical gradient computation for Woh
    E, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx = rnn.BackPropagate(input_idxs, target_idxs)
    
    epsilon = 1e-7
    baseWoh = np.copy(rnn.Woh)
    numdWoh = np.zeros([rnn.output_size, rnn.hidden_size],dtype=DTYPE)
    for i in range(rnn.output_size):
        for j in range(rnn.hidden_size):
            newWoh = np.copy(baseWoh)
            newWoh[i,j] += epsilon
            rnn.Woh = newWoh

            newE, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
            numdWoh[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWoh - dWoh)
    assert diff < 1e-3
    print 'Woh Check Passed! Diff is', diff
    
    # Numerical gradient computation for Whh
    E, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx = rnn.BackPropagate(input_idxs, target_idxs)
    
    epsilon = 1e-7
    baseWhh = np.copy(rnn.Whh)
    numdWhh = np.zeros([rnn.hidden_size, rnn.hidden_size],dtype=DTYPE)
    for i in range(rnn.hidden_size):
        for j in range(rnn.hidden_size):
            newWhh = np.copy(baseWhh)
            newWhh[i,j] += epsilon
            rnn.Whh = newWhh

            newE, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
            numdWhh[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWhh - dWhh)
    assert diff < 1e-3
    print 'Whh Check Passed! Diff is', diff
    
    # Numerical gradient computation for Whx
    E, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx = rnn.BackPropagate(input_idxs, target_idxs)
    
    epsilon = 1e-7
    baseWhx = np.copy(rnn.Whx)
    numdWhx = np.zeros([rnn.hidden_size, rnn.input_size],dtype=DTYPE)
    for i in range(rnn.hidden_size):
        for j in range(rnn.input_size):
            newWhx = np.copy(baseWhx)
            newWhx[i,j] += epsilon
            rnn.Whx = newWhx

            newE, probs = rnn.ForwardPropagate(input_idxs, target_idxs)
            numdWhx[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWhx - dWhx)
    assert diff < 1e-3
    print 'Whx Check Passed! Diff is', diff
