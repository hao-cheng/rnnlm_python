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

def eval_lm(model, batched_data, vocab):
    #model.CheckParams()

    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']
    append_idx = vocab['<append>']
    skip_set = [append_idx, unk_idx]

    total_logp = 0.0
    sents_processed = 0
    ivcount = 0
    oovcount = 0
    
    for batch_idx in range(len(batched_data['input_idxs'])):
        cur_batch_seq_inputs = batched_data['input_idxs'][batch_idx]
        cur_batch_seq_targets = batched_data['target_idxs'][batch_idx]
        model.ResetStates()

        curr_logp = 0.0
        for time_idx in range(len(cur_batch_seq_inputs)):
            input_idxs = cur_batch_seq_inputs[time_idx]
            target_idxs = cur_batch_seq_targets[time_idx]

            loss, probs, iv_count = model.ForwardPropagate(input_idxs, target_idxs, True, skip_set, True)
            curr_logp += loss
            ivcount += iv_count

        sents_processed += model.batch_size
        total_logp += curr_logp
        if (sents_processed % 200) == 0:
            print '.',
            sys.stdout.flush()
    
    print ' '
    print 'IV words: {}'.format(ivcount)
    print 'model perplexity: {}'.format(np.exp(-total_logp / ivcount))
    
    return total_logp

def batch_data(txt, vocab, batchsize=1, bptt=1):
    print '************************Data Preparation****************************'
    print 'bptt : {}'.format(bptt)
    print 'batchsize: {}'.format(batchsize)
    max_len = -1
    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']
    append_idx = vocab['<append>']
    for sent in txt:
        if max_len < len(sent.split()):
            max_len = len(sent.split())
    sent_in_idx = []
    max_len += 2
    for sent in txt:
        idx_words = []
        idx_words.append(eos_idx)
        count = 1
        for word in sent.split():
            if word in vocab:
                idx_words.append(vocab[word])
            else:
                idx_words.append(unk_idx)
            count += 1
        idx_words.append(eos_idx)
        count += 1
        while count < max_len:
            idx_words.append(append_idx)
            count += 1
        sent_in_idx.append(idx_words)

    time_idx = 0
    batch_idx = 0
    batch_data = {}
    total_sents = len(txt)
    batch_data['input_idxs'] = []
    batch_data['target_idxs'] = []
    for batch_idx in range(0, total_sents, batchsize):
        cur_inputs = []
        cur_targets = []
        for time_idx in range(max_len - bptt):
            input_idxs = []
            target_idxs = []
            for t in range(time_idx, time_idx + bptt):
                input_idx = []
                target_idx = []
                target_mult = []
                for b in range(batch_idx, batch_idx + batchsize):
                    if b >= total_sents:
                        input_idx.append(eos_idx)
                        target_idx.append(eos_idx)
                        target_mult.append(0.0)
                    else:
                        input_idx.append(sent_in_idx[b][t])
                        target_idx.append(sent_in_idx[b][t + 1])
                        target_mult.append(1.0)
                input_idxs.append(input_idx)
                target_idxs.append((target_idx, target_mult))
            cur_inputs.append(input_idxs)
            cur_targets.append(target_idxs)
        batch_data['input_idxs'].append(cur_inputs)
        batch_data['target_idxs'].append(cur_targets)
    print 'Data Preparation Done!'
    return batch_data

def batch_sgd_train(rnn_model, init_learning_rate, batch_size, train_txt, \
        valid_txt, outmodel, vocab, bptt, shuffle, tol):
    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']
    append_idx = vocab['<append>']
    skip_set = [append_idx, unk_idx]
    
    #train_txt = train_txt[:100]
    #valid_txt = valid_txt[:100]
    batched_data = batch_data(train_txt, vocab, batch_size, bptt)
    if valid_txt == []:
        batched_valid = None
    else:
        batched_valid = batch_data(valid_txt, vocab, batch_size)

    last_logp = -np.finfo(float).max
    curr_logp = last_logp

    sents_processed = 0
    iters = 0

    learning_rate = init_learning_rate
    #sent_indices = np.arange(len(train_txt))
    batch_indices = np.arange(len(batched_data['input_idxs']))
    #np.random.shuffle(batch_indices)
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
        if shuffle:
            np.random.shuffle(batch_indices)
        #for sent_idx in sent_indices:
        for batch_idx in batch_indices:
            cur_batch_seq_inputs = batched_data['input_idxs'][batch_idx]
            cur_batch_seq_targets = batched_data['target_idxs'][batch_idx]
            
            rnn_model.ResetStates()

            for time_idx in range(len(cur_batch_seq_inputs)):
                input_idxs = cur_batch_seq_inputs[time_idx]
                target_idxs = cur_batch_seq_targets[time_idx]

                loss, probs, iv_count = rnn_model.ForwardPropagate(input_idxs, target_idxs, True, skip_set)
                logp += loss
                dWhh, dWoh, dWhx = rnn_model.BackPropagate(input_idxs, target_idxs)
                rnn_model.UpdateWeight(dWhh, dWoh, dWhx)
                ivcount += iv_count

            sents_processed += batch_size

            if (sents_processed % 500) == 0:
                print '.',
                sys.stdout.flush()

        print ''
        print 'iv words in training: {}'.format(ivcount)
        print 'perlexity on training:{}'.format(np.exp(-logp / ivcount))
        print 'log-likelihood on training:{}'.format(logp)

        print 'epoch done!'

        #curr_logp = logp

        if batched_valid != None:
            print '-------------Validation--------------'
            curr_logp = eval_lm(rnn_model, batched_valid, vocab)
            print 'log-likelihood on validation: {}'.format(curr_logp)
            

        last_end_time = end_time
        end_time = time.time()
        print 'time elasped {} secs for this iteration out of {} secs in total.'.format(\
                end_time - last_end_time, end_time - start_time)

        obj_diff = curr_logp - last_logp
        if obj_diff < 0:
            print 'validation log-likelihood decrease; restore parameters'
            #halve_learning_rate = True
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
        if __profile__:
            return

def train_rnn_lm(args):
    vocab = load_vocab(args.vocabfile)
    train_txt = load_txt(args.trainfile)
    valid_txt = []
    if args.valid:
        valid_txt = load_txt(args.validfile)

    vocab_size = len(vocab)

    rnn_model = rnn.RNN()
    rnn_model.set_init_range(args.init_range)
    rnn_model.set_learning_rate(args.init_alpha)
    rnn_model.set_input_size(vocab_size)
    rnn_model.set_hidden_size(args.nhidden)
    rnn_model.set_output_size(vocab_size)
    rnn_model.set_bptt_unfold_level(args.bptt)
    rnn_model.set_batch_size(args.batchsize)

    rnn_model.AllocateModel()
    if args.inmodel == None:
        rnn_model.InitializeParemters()
    else:
        rnn_model.ReadModel(args.inmodel)
        
    #rnn_model.ResetLayers()
    print args
    if __profile__:
        pr = cProfile.Profile()
        pr.enable()
        batch_sgd_train(rnn_model, args.init_alpha, args.batchsize, train_txt, \
            valid_txt, args.outmodel, vocab, args.tol)
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
    else:
        batch_sgd_train(rnn_model, args.init_alpha, args.batchsize, train_txt, \
            valid_txt, args.outmodel, vocab, args.bptt, args.shuffle, args.tol)


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
    if args.valid and args.validfile == None:
        sys.stderr.write('Error: If valid, the validfile can not be None!')
        sys.exit(1)

    #np.__config__.show()
    train_rnn_lm(args)
