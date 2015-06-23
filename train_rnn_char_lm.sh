#!/usr/bin/env bash
#===============================================================================
#
#          FILE: train_rnn_char_lm.sh
# 
#         USAGE: ./train_rnn_char_lm.sh 
# 
#   DESCRIPTION: Train a char RNN language model
# 
#         NOTES: ---
#       CREATED: 06/20/15 19:19
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

trainfile=data/tinyshakespeare/train.txt
validfile=data/tinyshakespeare/valid.txt
vocabfile=data/tinyshakespeare/voc.txt
outmodel=expts/debug.20150620
nhidden=10
initalpha=1e-1
batchsize=1

python ./src/train_rnn_lm.py \
  --trainfile ${trainfile} \
  --validfile ${validfile} \
  --vocabfile ${vocabfile} \
  --init-alpha ${initalpha} \
  --batchsize ${batchsize} \
  --nhidden ${nhidden} \
  --outmodel ${outmodel} \
  --shuffle-sentence

