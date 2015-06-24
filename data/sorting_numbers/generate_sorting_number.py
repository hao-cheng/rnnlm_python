#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np

def generate_sorting_number(args):
    with open(args.outfn, 'w') as fout:
        seq = np.arange(args.low, args.high)
        seq_str = ' '.join(str(num) for num in seq)

        for i in range(args.size):
            pl = np.random.permutation(seq).tolist()
            pl_str = ' '.join(str(num) for num in pl)
            fout.write(\
                '{} <sort> {}\n'.format(pl_str, seq_str))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Generate random sorting number data')
    pa.add_argument('--low', type=int, default=0,\
            help='lower bound of number sequence')
    pa.add_argument('--high', type=int, default=5,\
            help='upper bound of number sequence')
    pa.add_argument('--size', type=int, default=100,\
            help='number of number sequences')
    pa.add_argument('--outfn', \
            help='output filename')
    args = pa.parse_args()
    generate_sorting_number(args)
    
