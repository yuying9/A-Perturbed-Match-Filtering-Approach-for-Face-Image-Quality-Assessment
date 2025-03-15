#!/usr/bin/env python
import argparse
import os
import glob
import math
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn import metrics

from collections import defaultdict
import gc
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from roc import get_eer_threshold

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--embeddings_dir', type=str,
                    default="../data/quality_embeddings/",
                    help='The dir save embeddings for each method and dataset, the diretory inside should be: {dataset}_{model}, e.g., IJBC_ArcFaceModel')
parser.add_argument('--quality_score_dir', type=str,
                    default="../data/quality_data/",
                    help='The dir save file of quality scores for each dataset and method, the file inside should be: {method}_{dataset}.txt, e.g., CRFIQAS_IJBC.txt')
parser.add_argument('--method_name', type=str,
                    default="ElasticFaceModel_PMF",
                    help='The evaluated image quality estimation method')
parser.add_argument('--models', type=str,
                    default="ArcFaceModel",
                    # default="ArcFaceModel, ElasticFaceModel, MagFaceModel, CurricularFaceModel,AdaFaceModel",
                    help='The evaluated FR model')
parser.add_argument('--eval_db', type=str,
                    default="lfw",
                    # default="lfw,cfp_fp,cplfw,calfw,agedb_30,XQLFW",
                    help='The evaluated dataset')
parser.add_argument('--distance_metric', type=str,
                    default='cosine',
                    help='Cosine distance or euclidian distance')
parser.add_argument('--output_dir', type=str,
                    default="erc_plot_auc_test_all",
                    help='')

IMAGE_EXTENSION = '.jpg'

def load_quality(scores):
    quality={}
    with open(scores[0], 'r') as f:
        lines=f.readlines()
        for l in lines:
            scores = l.split()[1].strip()
            n = "../data/quality_data/" + l.split()[0].strip()
            quality[n] = scores
    return quality

def load_quality_pair(pair_path, scores, dataset, args):
    pairs_quality = []
    quality=load_quality(scores)
    with open(pair_path, 'r') as f:
        lines=f.readlines()
        for idex in range(len(lines)):
            a= lines[idex].rstrip().split()[0]
            b= lines[idex].rstrip().split()[1]

            if dataset == 'XQLFW':
                qlt=min(float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{a}"))),
                    float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{b}"))))
            else:
                qlt=min(float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{a}{IMAGE_EXTENSION}"))),
                    float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{b}{IMAGE_EXTENSION}"))))

            pairs_quality.append(qlt)
    return pairs_quality

def load_feat_pair(pair_path, root):
    pairs = {}
    with open(pair_path, 'r') as f:
        lines=f.readlines()
        for idex in range(len(lines)):
            a= lines[idex].rstrip().split()[0]
            b= lines[idex].rstrip().split()[1]
            is_same=int(lines[idex].rstrip().split()[2])
            a = a.split('.')[0]
            b = b.split('.')[0]
            feat_a=np.load(os.path.join(root, f"{a}.npy"))
            feat_b=np.load(os.path.join(root, f"{b}.npy"))
            pairs[idex] = [feat_a, feat_b, is_same]
    print("All features are loaded")
    return pairs

def distance_(embeddings0, embeddings1, dist="cosine"):
    # Distance based on cosine similarity
    if (dist=="cosine"):
        dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
        norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
        # shaving
        similarity = np.clip(dot / norm, -1., 1.)
        dist = np.arccos(similarity) / math.pi
    else:
        embeddings0 = sklearn.preprocessing.normalize(embeddings0)
        embeddings1 = sklearn.preprocessing.normalize(embeddings1)
        diff = np.subtract(embeddings0, embeddings1)
        dist = np.sum(np.square(diff), 1)

    return dist

def calc_score(embeddings0, embeddings1, actual_issame, subtract_mean=False, dist_type='cosine'):
    assert (embeddings0.shape[0] == embeddings1.shape[0])
    assert (embeddings0.shape[1] == embeddings1.shape[1])

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings0, embeddings1]), axis=0)
    else:
        mean = 0.

    dist = distance_(embeddings0, embeddings1, dist=dist_type)
    # sort in a desending order
    pos_scores =np.sort(dist[actual_issame == 1])
    neg_scores = np.sort(dist[actual_issame == 0])
    return pos_scores, neg_scores

def calculate_pAUC(fnmrs_lists, method_labels, model, output_dir, fmr, db):

    unconsidered_rates = 100 * np.arange(0, 0.32, 0.05)

    for i in range(len(fnmrs_lists)):

        fnmrs_lists_3 = fnmrs_lists[i][:len(unconsidered_rates)]
        fnmrs_list_base = fnmrs_lists[i][0:1]*len(unconsidered_rates)
        # print(fnmrs_list_base)

        auc_value =  metrics.auc(np.array(unconsidered_rates/100), np.array(fnmrs_lists_3))
        auc_value_base = metrics.auc(np.array(unconsidered_rates/100), np.array(fnmrs_list_base))

        print(db, model, method_labels,'pAUC: %.2f' % (auc_value*1000),  'Normalized pAUC: %.3f' %(auc_value/auc_value_base))


def getFNMRFixedTH(feat_pairs, qlts,  dist_type='cosine', desc=True):
    embeddings0, embeddings1, targets = [], [], []
    pair_qlt_list = []  # store the min qlt
    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])
        # convert into np
        np_feat_a = np.asarray(feat_a, dtype=np.float64)
        np_feat_b = np.asarray(feat_b, dtype=np.float64)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)
        targets.append(ab_is_same)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1, )
    qlts = np.array(qlts)
    if (desc):
        qlts_sorted_idx = np.argsort(qlts)
    else:
        qlts_sorted_idx = np.argsort(qlts)[::-1]

    num_pairs = len(targets)
    # unconsidered_rates = np.arange(0, 0.98, 0.05)
    unconsidered_rates = np.arange(0, 0.32, 0.05)

    fnmrs_list = []

    for u_rate in unconsidered_rates:
        hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs):]
        pos_dists, neg_dists = calc_score(embeddings0[hq_pairs_idx], embeddings1[hq_pairs_idx], targets[hq_pairs_idx], dist_type=dist_type)
        fmr1000_th = get_eer_threshold(pos_dists, neg_dists, ds_scores=True)

        g_true = [g for g in pos_dists if g < fmr1000_th]
        fnmrs_list.append(1 - len(g_true) / (len(pos_dists)))

    return fnmrs_list, unconsidered_rates



def perform_1v1_quality_eval(args):
 d = args.eval_db.split(',')


 models=args.models.split(',')
 for model in models:
  for dataset in d:
    method_names = args.method_name.split(',')
    method_labels=[]

    if (not os.path.isdir(os.path.join(args.output_dir, dataset, 'fnmr'))):
        os.makedirs(os.path.join(args.output_dir, dataset, 'fnmr'))

    for method_name in method_names:
        fnmrs_list=[]

        print(f"----process {model} {dataset} {method_name}-----------")
        desc = False if method_name == 'PFE' else True

        print('load feat pairs ......')
        feat_pairs = load_feat_pair(os.path.join(args.quality_score_dir, dataset, 'pair_list.txt'),
                                   os.path.join(args.embeddings_dir, f"{dataset}_{model}"))


        quality_scores = load_quality_pair(os.path.join(args.quality_score_dir, dataset, 'pair_list.txt'),
                                          [os.path.join('../data/quality_score/', f"{dataset}_{method_name}.txt")],
                                          dataset, args)


        fnmr3, unconsidered_rates = getFNMRFixedTH(feat_pairs, quality_scores, dist_type=args.distance_metric, desc=desc)
        fnmrs_list.append(fnmr3)
        method_labels.append(f"{method_name}")

        calculate_pAUC(fnmrs_list, method_name, model=model, output_dir=args.output_dir, fmr =1e-3, db=dataset)


def main():
    args = parser.parse_args()
    perform_1v1_quality_eval(args)

if __name__ == '__main__':
    main()
