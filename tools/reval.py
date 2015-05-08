#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

import _init_paths
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
import cPickle
import os, sys, argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to re-evaluate',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def reval_dets(imdb_name, output_dir, comp_mode):
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(comp_mode)
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = cPickle.load(f)

    print 'Evaluating detections'
    imdb.evaluate_detections(dets, output_dir)

if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir[0])
    imdb_name = args.imdb_name

    reval_dets(imdb_name, output_dir, args.comp_mode)
