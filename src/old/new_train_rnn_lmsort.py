#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import new_neuralnet.rnn as rnn

## profile helpers
import cProfile, pstats, StringIO
#__profile__ = True
__profile__ = False

from new_train_rnn_lm import train_rnn_lm

def batch_data(txt, vocab, batch_size=1, bptt=1):
    print '************************Data Preparation****************************'
    print 'bptt : {}'.format(bptt)
    print 'batch_size: {}'.format(batch_size)
    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']
    append_idx = 0
    sent_in_idx = []
    for sent in txt:
        idx_words = []
        sep_seen = 0.0
        idx_words.append((eos_idx,sep_seen))
        for word in sent.split():
            if word in vocab:
                idx_words.append((vocab[word],sep_seen))
                if word == '<sort>':
                    sep_seen = 1.0
            else:
                assert False
                idx_words.append((unk_idx,sep_seen))
        idx_words.append((eos_idx,sep_seen))
        sent_in_idx.append(idx_words)

    # Shuffle examples
    np.random.shuffle(sent_in_idx)
    batch_data = {}
    batch_data['input_idxs'] = []
    batch_data['target_idxs'] = []
    in_flight_sents = [[]] * batch_size
    done = False
    while not done:
        cur_inputs = []
        cur_targets = []
        done = True
        for b in range(batch_size):
            if len(in_flight_sents[b]) <= 1:
                if len(sent_in_idx) > 0:
                    in_flight_sents[b] = sent_in_idx[0]
                    del sent_in_idx[0]
                    done = False
            else:
                done = False
        if done:
            break
            
        for t in range(bptt):
            cur_input = [0] * batch_size
            cur_target = [0] * batch_size
            cur_weight = [0.0] * batch_size
            for b in range(batch_size):
                sent = in_flight_sents[b]
                if (len(sent) >= 2):
                    cur_input[b] = sent[0][0]
                    cur_target[b] = sent[1][0]
                    cur_weight[b] = sent[1][1]
                    in_flight_sents[b] = sent[1:]
                else:
                    #Done with this
                    in_flight_sents[b] = []
            cur_inputs.append(cur_input)
            cur_targets.append((cur_target, cur_weight))

        batch_data['input_idxs'].append(cur_inputs)
        batch_data['target_idxs'].append(cur_targets)
    print 'Data Preparation Done!'
    return batch_data

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    ## Require arguments 
    pa.add_argument('--trainfile', \
            help='train filename (REQUIRED)')
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--outmodel', \
            help='output model name (REQUIRED)')
    ## Optional arguments
    pa.add_argument('--validfile', \
            help='validation filename')
    pa.add_argument('--validate', action='store_true', \
            dest='valid', \
            help='validation during training')
    pa.set_defaults(valid=False)
    pa.add_argument('--inmodel', \
            help='input model name')
    pa.add_argument('--nhidden', type=int, default=10, \
            help='hidden layer size, integer > 0')
    pa.add_argument('--bptt', type=int, default=12, \
            help='backpropagate through time level, integer >= 1')
    pa.add_argument('--init-alpha', type=float, default=0.1, \
            help='initial learning rate, scalar > 0')
    pa.add_argument('--init-range', type=float, default=0.1, \
            help='random initial range, scalar > 0')
    pa.add_argument('--batchsize', type=int, default=1, \
            help='training batch size, integer >= 1')
    pa.add_argument('--tol', type=float, default=1e-3, \
            help='minimum improvement for log-likelihood, scalar > 0')
    pa.add_argument('--shuffle-sentence', action='store_true', 
            dest='shuffle', \
            help='shuffle sentence every epoch')
    pa.set_defaults(shuffle=False)
    args = pa.parse_args()

    if args.trainfile == None or \
        args.vocabfile == None or \
        args.outmodel == None or \
        args.nhidden < 1 or \
        args.init_alpha <= 0 or \
        args.init_range <= 0 or \
        args.bptt < 1 or \
        args.batchsize < 1 or \
        args.tol <= 0.0:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)
    if args.valid and args.validfile == None:
        sys.stderr.write('Error: If valid, the validfile can not be None!')
        sys.exit(1)

    #np.__config__.show()
    train_rnn_lm(args)
