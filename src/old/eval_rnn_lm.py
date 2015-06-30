#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import neuralnet.recurrent_neuralnet as rnn

## profile helpers
import cProfile, pstats, StringIO
#__profile__ = True
__profile__ = False

def load_vocab(fn):
    vocab = {}
    word_idx = 0
    with open(fn) as fin:
        for line in fin:
            word = line.strip()
            vocab[word] = word_idx
            word_idx += 1
    if '</s>' not in vocab:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in vocab:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return vocab

def load_txt(fn):
    txt = []
    with open(fn) as fin:
        for line in fin:
            txt.append(line.strip())
    return txt

def eval_lm(model, txt, vocab):
    model.CheckParams()

    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']

    total_logp = 0.0
    sents_processed = 0
    ivcount = 0
    oovcount = 0

    for sent in txt:
        model.ResetLayers()

        words = sent.strip().split(' ')
        curr_logp = 0.0
        model.ForwardPropagate(eos_idx)
        for word in words:
            word_idx = unk_idx
            if word not in vocab:
                oovcount += 1
            else:
                ivcount += 1
                word_idx = vocab[word]
                curr_logp += model.ComputeLogProb(word_idx)
            model.ForwardPropagate(word_idx)
        curr_logp += model.ComputeLogProb(eos_idx)
        ivcount += 1

        total_logp += curr_logp
        sents_processed += 1
        if np.mod(sents_processed, 200) == 0:
            print '.',
            sys.stdout.flush()
    
    if ivcount == 0:
        sys.stderr.write('Error: zero IV word!\n')
        sys.exit(1)

    print ' '
    print 'num of IV words : {}'.format(ivcount)
    print 'num of OOV words: {}'.format(oovcount)
    print 'model perplexity: {}'.format(np.exp(-total_logp / ivcount))
    
    return total_logp

def eval_rnn_lm(args):
    vocab = load_vocab(args.vocabfile)
    test_txt = load_txt(args.testfile)

    vocab_size = len(vocab)

    rnn_model = rnn.RecurrentNeuralNet()

    if args.inmodel == None:
        rnn_model.AllocateModel()
        rnn_model.InitializeNeuralNet()
    else:
        rnn_model.ReadModel(args.inmodel)
    eval_lm(rnn_model, test_txt, vocab)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    ## Require arguments 
    pa.add_argument('--testfile', \
            help='test filename (REQUIRED)')
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel', \
            help='inmodel name (REQUIRED)')
    args = pa.parse_args()

    if args.testfile == None or \
        args.vocabfile == None or \
        args.inmodel == None:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)

    eval_rnn_lm(args)

