#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import new_neuralnet.rnn as rnn

def build_idx_word(fn):
    word4idx = {}
    idx4word ={}
    word_idx = 0
    with open(fn) as fin:
        for line in fin:
            word = line.strip()
            idx4word[word] = word_idx
            word4idx[word_idx] = word
            word_idx += 1
    if '</s>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return idx4word, word4idx 

def get_most_prob_word(ch_idx, model, word4idx, idx4word):
    sep_idx = idx4word['<sep>']
    append_idx = idx4word['<append>']
    word = []
    word.append(word4idx[ch_idx])
    model.set_batch_size(1)
    model.set_bptt_unfold_level(1)
    model.ResetStates()
    cur_ch_idx = ch_idx
    while True:
        input_idx = []
        input_idx.append([cur_ch_idx])
        target_idx = []
        target_idx.append(([sep_idx], [0.0]))
        _, probs = model.ForwardPropagate(input_idx, target_idx)
        # rule out append
        probs[0][append_idx] = 0
        idx = np.argmax(probs[0])
        if idx == sep_idx:
            break
        word.append(word4idx[idx])
        cur_ch_idx = idx

    return ''.join(word)

def get_most_prob_sent(word, model, word4idx, idx4word):
    sep_idx = idx4word['<sep>']
    append_idx = idx4word['<append>']
    eos_idx = idx4word['</s>']
    sent = []
    sent.append(word)
    model.set_batch_size(1)
    model.set_bptt_unfold_level(1)
    model.ResetStates()
    probs = []
    chs = ' '.join(word).split()
    chs.insert(0, '</s>')
    print chs
    for ch in chs:
        idx = idx4word[ch]
        input_idx = []
        input_idx.append([idx])
        target_idx = []
        target_idx.append(([sep_idx], [0.0]))

        _, probs = model.ForwardPropagate(input_idx, target_idx)

    max_len = 10
    word = []
    while True:
        idx = np.argmax(probs[0])
        if idx == eos_idx or idx == append_idx:
            sent.append(''.join(word))
            break
        if idx == sep_idx:
            sent.append(''.join(word))
            word = []
        else:
            word.append(word4idx[idx])
        if len(sent) >= max_len or \
                len(word) >= max_len:
            break

        input_idx = []
        input_idx.append([idx])
        target_idx = []
        target_idx.append(([sep_idx], [0.0]))
        _, probs = model.ForwardPropagate(input_idx, target_idx)
        # rule out append
        probs[0][append_idx] = 0

    return ' '.join(sent)


def sample_rnn_lm(args):
    idx4word, word4idx = build_idx_word(args.vocabfile)
    sep_idx = idx4word['<sep>']

    rnn_model = rnn.RNN()
    rnn_model.ReadModel(args.inmodel)

    print 'Play with RNN Language Model on Tiny Shakespeare, To exit enter 0'
    if args.sample_sent:
        # sample sent
        print 'Generate sentence from RNN LM'
        while True:
            first_word = raw_input('Enter the word (lowercased) to start with: ')
            if first_word == '0':
                break
            sent = get_most_prob_sent(first_word, rnn_model, word4idx, idx4word)
            print 'The most probable sentence starting with {} is: \n{}'.format(\
                    first_word, sent)
        
    elif args.spell_word:
        # sample word 
        print 'Spell word from RNN LM'
        while True:
            first_char = raw_input('Enter the character (a-z) to start with: ')
            if first_char == '0':
                break
            if first_char not in idx4word:
                print 'Input {} is not a valid character'.format(first_char)
                continue
            print 'Entered: {}'.format(first_char)
            word = get_most_prob_word(idx4word[first_char], rnn_model, word4idx, idx4word)
            print 'The most probable word starting with {} is: {}'.format(\
                    first_char, word)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    ## Require arguments 
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel', \
            help='inmodel name (REQUIRED)')
    pa.add_argument('--spell-word', action='store_true',\
            dest='spell_word', help='sample LM to spell word')
    pa.set_defaults(spell_word=False)
    pa.add_argument('--sample-sent', action='store_true',\
            dest='sample_sent', help='sample LM to generate sentence')
    pa.set_defaults(sample_sent=False)
    pa.add_argument('--compute-logp', action='store_true',\
            dest='compute_logp', help='compute logprob of given word')
    args = pa.parse_args()

    if args.vocabfile == None or \
        args.inmodel == None:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)

    sample_rnn_lm(args)

