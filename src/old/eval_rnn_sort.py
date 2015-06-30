#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import neuralnet.recurrent_neuralnet as rnn

def load_map(fn):
    idx4word = {}
    word4idx = {}
    word_idx = 0
    with open(fn) as fin:
        for line in fin:
            word = line.strip()
            idx4word[word] = word_idx
            word4idx[word_idx] = word
            word_idx += 1
    if '</s>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
        sys.exit(1)
    if '<unk>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
        sys.exit(1)
    return idx4word, word4idx 

def load_txt(fn):
    txt = []
    with open(fn) as fin:
        for line in fin:
            txt.append(line.strip())
    return txt

def eval_sort(model, txt, idx4word, word4idx, outfn):
    model.CheckParams()

    eos_idx = idx4word['</s>']
    unk_idx = idx4word['<unk>']

    total_logp = 0.0
    acc = 0.0
    sents_processed = 0
    ivcount = 0
    oovcount = 0
    count = 0

    seqs = {}
    seqs['input'] = []
    seqs['hyp'] = []
    seqs['truth'] = []

    print '*****************************Evaluating***************************'
    for sent in txt:
        model.ResetLayers()
        words = sent.strip().split(' ')
        curr_logp = 0.0
        eval_logp = False
        model.ForwardPropagate(eos_idx)
        input_seq = []
        true_seq = []
        new_seq = []
        new_indices = set([])
        new_indices.add(eos_idx)
        new_indices.add(unk_idx)
        new_indices.add(idx4word['<sort>'])
        item_set = len(idx4word) * [False]
        for word in words:
            word_idx = unk_idx
            if word not in idx4word:
                oovcount += 1
            else:
                ivcount += 1
                word_idx = idx4word[word]
            if eval_logp:
                curr_logp += model.ComputeLogProb(word_idx)
                true_idx = word_idx
                true_seq.append(word)
                #word_idx = model.GetMostProbNext()
                #word_idx = model.GetMostProbUniqNext(new_indices)
                word_idx = model.GetMostProbUniqNextFromSet(new_indices,\
                        item_set)
                if true_idx == word_idx:
                    acc += 1
                new_seq.append(word4idx[word_idx])
                new_indices.add(word_idx)
                count += 1
            else:
                if word == '<sort>':
                    eval_logp = True
                else:
                    input_seq.append(word4idx[word_idx])
                    item_set[word_idx] = True
            model.ForwardPropagate(word_idx)
        #curr_logp += model.ComputeLogProb(eos_idx)
        #ivcount += 1


        seqs['input'].append(' '.join(input_seq))
        seqs['hyp'].append(' '.join(new_seq))
        seqs['truth'].append(' '.join(true_seq))

        total_logp += curr_logp
        sents_processed += 1
        if np.mod(sents_processed, 200) == 0:
            print '.',
            sys.stdout.flush()
    
    if ivcount == 0:
        sys.stderr.write('Error: zero IV word!\n')
        sys.exit(1)

    for i in range(len(seqs['hyp'])):
        print 'Test Example {}'.format(i)
        print 'input seq: {}'.format(seqs['input'][i])
        print 'hyp seq: {}'.format(seqs['hyp'][i])
        print 'true seq: {}'.format(seqs['truth'][i])
        print ''


    print 'num of IV words : {}'.format(ivcount)
    print 'num of OOV words: {}'.format(oovcount)
    print 'Accuracy: {}%'.format('%.1f' % (acc / count * 100))
    print 'loglikelihood :{}'.format(total_logp)

    
    #return total_logp / ivcount
    return total_logp

def eval_rnn_sort(args):
    #vocab = load_vocab(args.vocabfile)
    idx4word, word4idx = load_map(args.vocabfile)
    eval_txt = load_txt(args.infile)

    vocab_size = len(idx4word)

    rnn_model = rnn.RecurrentNeuralNet()

    rnn_model.ReadModel(args.inmodel)
    rnn_model.AllocateModel()
    rnn_model.ResetLayers()
    print args
    eval_sort(rnn_model, eval_txt, idx4word, word4idx, args.outfile)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN sorting model')
    ## Require arguments 
    pa.add_argument('--infile', \
            help='eval filename (REQUIRED)')
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel', \
            help='input model name (REQUIRED)')
    ## Optional arguments
    pa.add_argument('--outfile', \
            help='output filename')
    args = pa.parse_args()

    if args.infile == None or \
        args.vocabfile == None or \
        args.inmodel == None:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)

    #np.__config__.show()
    eval_rnn_sort(args)
