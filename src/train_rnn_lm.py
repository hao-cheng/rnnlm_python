#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import neuralnet.recurrent_neuralnet as rnn

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
    
    if ivcount == 0:
        sys.stderr.write('Error: zero IV word!\n')
        sys.exit(1)

    print ' '
    print 'num of IV words : {}'.format(ivcount)
    print 'num of OOV words: {}'.format(oovcount)
    print 'model perplexity: {}'.format(np.exp(-total_logp / ivcount))
    
    return total_logp / ivcount



def batch_sgd_train(rnn_model, init_learning_rate, batch_size, train_txt, \
        valid_txt, outmodel, vocab, tol):
    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']

    last_logp = -np.finfo(float).max
    curr_logp = last_logp

    sents_processed = 0
    iters = 0

    learning_rate = init_learning_rate
    sent_indices = np.arange(len(train_txt))
    np.random.shuffle(sent_indices)
    halve_learning_rate = False

    start_time = time.time()
    end_time = start_time
    last_end_time = end_time
    while True:
        iters += 1
        print '******************************* Iteration {} ***********************************'.format(iters)

        rnn_model.set_learning_rate(learning_rate)
        print 'learning_rate = {}\n'.format(learning_rate)

        logp = 0.0
        ivcount = 0;
        oovcount = 0;
        batch_count = 0

        for sent_idx in sent_indices:
            words = train_txt[sent_idx].split()
            rnn_model.ResetLayers()

            rnn_model.ForwardPropagate(eos_idx)
            for word in words:
                word_idx = unk_idx
                if word not in vocab:
                    oovcount += 1
                else:
                    ivcount += 1
                    word_idx = vocab[word]
                    logp += rnn_model.ComputeLogProb(word_idx)
                rnn_model.BackPropagate(word_idx)
                batch_count += 1
                if (batch_count == batch_size):
                    rnn_model.UpdateWeight(learning_rate)
                    batch_count = 0
                rnn_model.ForwardPropagate(word_idx)
            ivcount += 1
            logp += rnn_model.ComputeLogProb(eos_idx)
            rnn_model.BackPropagate(eos_idx)
            if (batch_count == batch_size):
                rnn_model.UpdateWeight(learning_rate)
                batch_count = 0
            
            sents_processed += 1

            if np.mod(sents_processed, 500) == 0:
                print '.',
        rnn_model.UpdateWeight(learning_rate)
        batch_count = 0

        print '\n'
        print 'num IV words in training: {}'.format(ivcount)
        print 'num OOV words in training: {}'.format(oovcount)

        print 'model perplexity on training:{}'.format(\
                np.exp(-logp / ivcount))
        print 'log-likelihood on training:{}'.format(logp / ivcount)

        print 'epoch done!'

        if valid_txt == []:
            # not validation during training
            curr_logp = logp / ivcount
        else:
            print '-------------Validation--------------'
            curr_logp = eval_lm(rnn_model, valid_txt, vocab)
            print 'log-likelihood on validation: {}'.format(curr_logp)
            


        last_end_time = end_time
        end_time = time.time()
        print 'time elasped {} secs for this iteration out of {} secs in total.'.format(\
                end_time - last_end_time, end_time - start_time)

        obj_diff = curr_logp - last_logp
        if obj_diff < 0:
            print 'validation log-likelihood decrease; restore parameters'
            rnn_model.RestoreModel()
        else:
            rnn_model.CacheModel()

        if obj_diff <= tol:
            if not halve_learning_rate:
                halve_learning_rate = True
            else:
                if outmodel != '':
                    rnn_model.WriteModel(outmodel)
                break

        if halve_learning_rate:
            learning_rate *= 0.5
        last_logp = curr_logp

def train_rnn_lm(args):
    vocab = load_vocab(args.vocabfile)
    train_txt = load_txt(args.trainfile)
    valid_txt = []
    if args.valid:
        valid_txt = load_txt(args.validfile)

    vocab_size = len(vocab)

    rnn_model = rnn.RecurrentNeuralNet()
    rnn_model.set_init_range(args.init_range)
    rnn_model.set_learning_rate(args.init_alpha)
    rnn_model.set_input_size(vocab_size)
    rnn_model.set_hidden_size(args.nhidden)
    rnn_model.set_output_size(vocab_size)
    rnn_model.set_bptt_unfold_level(args.bptt)

    if args.inmodel == None:
        rnn_model.AllocateModel()
        rnn_model.InitializeNeuralNet()
    else:
        rnn_model.ReadModel(args.inmodel)
        rnn_model.AllocateModel()
    rnn_model.ResetLayers()
    print args
    batch_sgd_train(rnn_model, args.init_alpha, args.batchsize, train_txt, \
        valid_txt, args.outmodel, vocab, args.tol)


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
    pa.add_argument('--bptt', type=int, default=1, \
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

    #np.__config__.show()
    train_rnn_lm(args)
