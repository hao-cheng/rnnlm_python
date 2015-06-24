#!/usr/bin/env bash
#===============================================================================
#
#          FILE: eval_rnn_sort.sh
# 
#         USAGE: ./eval_rnn_sort.sh 
# 
#   DESCRIPTION: Evaluate a trained RNN sorting model
# 
#         NOTES: ---
#       CREATED: 06/20/15 19:19
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

datadir=data/sorting_numbers
infile=${datadir}/test
vocabfile=${datadir}/vocab
inmodel=expts/debug.sort.20150620
outfn=expts/debug.sort.20150620

python ./src/eval_rnn_sort.py \
  --infile ${infile} \
  --vocabfile ${vocabfile} \
  --inmodel ${inmodel} \
  --outfile ${outfn}

