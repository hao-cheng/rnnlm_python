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

#trainfile=data/tinyshakespeare/train.txt
#validfile=data/tinyshakespeare/valid.txt
testfile=data/tinyshakespeare/test.txt
#testfile=data/tinyshakespeare/valid.txt
vocabfile=data/tinyshakespeare/voc.txt
inmodel=expts/debug.char-lm.20150620

python ./src/eval_rnn_lm.py\
  --testfile ${testfile} \
  --vocabfile ${vocabfile} \
  --inmode ${inmodel}

