#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import cPickle as cp

import neuralnet_layer as layer

DTYPE = np.float32

class RecurrentNeuralNet():
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

        self.input_hidden_connection = None
        self.recurrent_hidden_connection = None
        self.hidden_output_connection = None
        self.last_input_hidden_connection = None
        self.last_recurrent_hidden_connection = None
        self.last_hidden_output_connection = None
        
        self.input_hidden_connection_grad = None
        self.recurrent_hidden_connection_grad = None
        self.hidden_output_connection_grad = None

        self.input_layers = None
        self.hidden_layers = None
        self.output_layer = None


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
    def CheckParams(self):
        ## Check parameters
        assert(self.input_size > 0)
        assert(self.hidden_size > 0)
        assert(self.output_size > 0)
        assert(self.init_range > 0)


    def InitializeNeuralNet(self):
        ## Randomly initialize the connection weights
        self.input_hidden_connection += np.random.uniform(-self.init_range, \
                self.init_range, [self.input_size, self.hidden_size])
        self.recurrent_hidden_connection += np.random.uniform(-self.init_range, \
                self.init_range, [self.hidden_size, self.hidden_size])
        self.hidden_output_connection += np.random.uniform(-self.init_range, \
                self.init_range, [self.hidden_size, self.output_size])

    def AllocateModel(self):
        ## Allocate model parameters
        self.input_hidden_connection = np.zeros([self.input_size, self.hidden_size], \
               dtype=DTYPE)
        self.recurrent_hidden_connection = np.zeros([self.hidden_size, self.hidden_size], \
               dtype=DTYPE)
        self.hidden_output_connection = np.zeros([self.hidden_size, self.output_size], \
                    dtype=DTYPE)

        self.last_input_hidden_connection = np.zeros([self.input_size, self.hidden_size], \
                    dtype=DTYPE)
        self.last_recurrent_hidden_connection = np.zeros([self.hidden_size, self.hidden_size], \
                    dtype=DTYPE)
        self.last_hidden_output_connection = np.zeros([self.hidden_size, self.output_size], \
                    dtype=DTYPE)

        self.input_hidden_connection_grad = np.zeros([self.input_size, self.hidden_size], \
                    dtype=DTYPE)
        self.recurrent_hidden_connection_grad = np.zeros([self.hidden_size, self.hidden_size], \
                    dtype=DTYPE)
        self.hidden_output_connection_grad = np.zeros([self.hidden_size, self.output_size], \
                    dtype=DTYPE)

        self.input_layers = layer.NeuralNetLayer(is_input =True)
        self.input_layers.set_size(self.input_size)
        self.input_layers.set_capacity(self.bptt_unfold_level)
        self.input_layers.AllocateLayers()
        self.hidden_layers = layer.NeuralNetLayer()
        self.hidden_layers.set_size(self.hidden_size)
        self.hidden_layers.set_capacity(self.bptt_unfold_level + 1)
        self.hidden_layers.AllocateLayers()
        self.output_layer = layer.NeuralNetLayer(is_output=True)
        self.output_layer.set_size(self.output_size)
        self.output_layer.AllocateLayers()

    def ResetLayers(self):
        ## Reset all layers
        self.input_layers.ResetLayer()
        self.hidden_layers.ResetLayer()
        self.output_layer.ResetLayer()

    def ResetLayerActivations(self):
        ## Reset layer activations
        self.input_layers.ResetActivations(0)
        self.hidden_layers.ResetActivations(0)
        self.output_layer.ResetActivations(0)

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

            self.input_hidden_connection = model['input_hidden_connection']
            self.recurrent_hidden_connection = model['recurrent_hidden_connection']
            self.hidden_output_connection = model['hidden_output_connection']
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

        model['input_hidden_connection'] = self.input_hidden_connection
        model['recurrent_hidden_connection'] = self.recurrent_hidden_connection
        model['hidden_output_connection'] = self.hidden_output_connection

        with open(fname, 'wb') as fout:
            print '=========Writing Model========\n'
            cp.dump(model, fout)
            print '=========Writing Done========\n'


    def CacheModel(self):
        ## Cache current model
        self.last_input_hidden_connection[:] = self.input_hidden_connection
        self.last_recurrent_hidden_connection[:] = self.recurrent_hidden_connection
        self.last_hidden_output_connection[:] = self.hidden_output_connection

    def RestoreModel(self):
        ## Restore previous model
        self.input_hidden_connection[:] = self.last_input_hidden_connection
        self.recurrent_hidden_connection[:] = self.last_recurrent_hidden_connection
        self.hidden_output_connection[:] = self.last_hidden_output_connection


    '''
    Forward propogation
    '''
    def ForwardPropagate(self, input_idx):
        self.input_layers.Rotate()
        self.hidden_layers.Rotate()
        self.ResetLayerActivations()

        current_input_layer_activations = self.input_layers.activation(0)
        current_input_layer_activations[input_idx] = 1.0
        current_hidden_layer_activations = self.hidden_layers.activation(0)
        prev_hidden_layer_activations = self.hidden_layers.activation(1)

        output_layer_activations = self.output_layer.activation(0)
        
        # current input layer -> current hidden layer
        current_hidden_layer_activations += \
                np.dot(self.input_hidden_connection.T, \
                        current_input_layer_activations)

        # prev hidden layer -> current hidden layer
        current_hidden_layer_activations += \
                np.dot(self.recurrent_hidden_connection.T, \
                        prev_hidden_layer_activations)

        # compute activation of the hidden layer
        # only update the current hidden layer
        self.hidden_layers.SigmoidActivation(current_hidden_layer_activations)

        # current hidden layer -> output layer
        output_layer_activations += \
                np.dot(self.hidden_output_connection.T, \
                        current_hidden_layer_activations)


        self.output_layer.SoftmaxActivation(output_layer_activations)
        

    '''
    Backpropogation through time
    '''
    def BackPropagate(self, output_idx):
        output_layer_activations = self.output_layer.activation(0)
        output_layer_errors = self.output_layer.error(0)

        # compute gradients and errors of the output layer
        self.output_layer.ResetErrors(0)
        self.output_layer.SoftmaxGradient(output_layer_errors, \
                output_layer_activations, output_idx)

        # backpropagate from output layer -> current hidden layer
        current_hidden_layer_activations = self.hidden_layers.activation(0)
        current_hidden_layer_errors = self.hidden_layers.error(0)

        # compute gradients and errors
        self.hidden_layers.ResetErrors(0)
        current_hidden_layer_errors += np.dot(self.hidden_output_connection, \
                output_layer_errors)
        self.hidden_layers.SigmoidGradient(current_hidden_layer_errors, \
                current_hidden_layer_activations)
        # backpropagate through time
        # Do not backpropagate to the oldest hidden layer
        i = 1
        while (i < self.bptt_unfold_level):
            # traverse back to time
            # backpropagate from current hidden layer -> prev hidden layer
            current_hidden_layer_activations = self.hidden_layers.activation(i - 1)
            current_hidden_layer_errors = self.hidden_layers.error(i - 1)

            prev_hidden_layer_activations = self.hidden_layers.activation(i)
            prev_hidden_layer_errors = self.hidden_layers.error(i)

            self.hidden_layers.ResetActivations(i)

            prev_hidden_layer_errors += np.dot(self.recurrent_hidden_connection,\
                    current_hidden_layer_errors)

            self.hidden_layers.SigmoidGradient(prev_hidden_layer_errors, \
                    prev_hidden_layer_activations)
            i += 1

        # no need to backpropagate to input layer

        ########################################
        ### Accumulate gradient of weights#####
        #######################################

        for i in range(self.bptt_unfold_level):
            ## Accumulate gradients input_hidden_connection
            current_input_activations = self.input_layers.activation(i)
            current_hidden_errors = self.hidden_layers.error(i)
            self.input_hidden_connection_grad += np.dot(current_input_activations, \
                    current_hidden_errors.T)

            ## Accmulate gradients recurrent_hidden_connection
            prev_hidden_activations = self.hidden_layers.activation(i + 1)
            self.recurrent_hidden_connection_grad += np.dot(prev_hidden_activations,\
                    current_hidden_errors.T)

        ## Accumulate gradients hidden_output_connection
        current_hidden_layer_activations = self.hidden_layers.activation(0)
        output_layer_errors = self.output_layer.error(0)
        self.hidden_output_connection_grad += np.dot(current_hidden_layer_activations, \
                output_layer_errors.T)
            

    '''
    Update connection weights
    '''
    def UpdateWeight(self, learning_rate):
        self.input_hidden_connection_grad *= learning_rate 
        self.input_hidden_connection += self.input_hidden_connection_grad
        self.recurrent_hidden_connection_grad *= learning_rate
        self.recurrent_hidden_connection += self.recurrent_hidden_connection_grad
        self.hidden_output_connection_grad *= learning_rate
        self.hidden_output_connection += self.hidden_output_connection_grad
        # reset gradients after update
        self.input_hidden_connection_grad[:] = 0
        self.recurrent_hidden_connection_grad[:] = 0
        self.hidden_output_connection_grad[:] = 0

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
