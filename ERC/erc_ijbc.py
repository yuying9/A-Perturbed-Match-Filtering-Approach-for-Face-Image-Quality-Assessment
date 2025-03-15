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

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from roc import get_eer_threshold

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--embeddings_dir', type=str,
                    default="../data/quality_embeddings",
                    help='The dir save embeddings for each method and dataset, the diretory inside should be: {dataset}_{model}, e.g., IJBC_ArcFaceModel')
parser.add_argument('--quality_score_dir', type=str,
                    default="../data/quality_score",
                    help='The dir save file of quality scores for each dataset and method, the file inside should be: {dataset}_{method}.txt, e.g., IJBC_ElasticFaceModel_PMF.txt')

parser.add_argument('--models', type=str,
                    default="ArcFaceModel",
                    help='The evaluated FR model')
parser.add_argument('--eval_db', type=str,
                    default="IJBC",
                    help='The evaluated dataset')
parser.add_argument('--output_dir', type=str,
                    default="erc_plot_auc_test_all",
                    help='')
parser.add_argument('--method_name', type=str,
                    default="ElasticFaceModel_PMF",
                    help='The evaluated image quality estimation method')

parser.add_argument('--distance_metric', type=str,
                    default='cosine',
                    help='Euclidian Distance or Cosine Distance.')
parser.add_argument('--feat_size', type=int,
                    default=1024,
                    help='The size of extracted features')

def load_all_features(root):
    all_features = defaultdict()
    for feature_path in tqdm(glob.glob(os.path.join(root, '*.npy'))):
        feat = np.load(feature_path)
        all_features[os.path.basename(feature_path)] = feat
    print("All features are loaded")
    print(len(all_features.keys()))
    print(all_features.keys())
    return all_features

def load_ijbc_pairs_features(pair_path, all_features, hq_pairs_idx, feature_size=1024):
    with open(pair_path, 'r') as f:
        lines=f.readlines()

    # build two empty embeddings matrix
    embeddings_0, embeddings_1 = np.empty([hq_pairs_idx.shape[0], feature_size]), np.empty([hq_pairs_idx.shape[0], feature_size])
    # load embeddings based on the needed pairs
    for indx in tqdm(range(hq_pairs_idx.shape[0])):
        real_index = hq_pairs_idx[indx]
        split_line = lines[real_index].split()
        feat_a = all_features[(split_line[0] + '.npy')]
        feat_b = all_features[(split_line[1] + '.npy')]
        embeddings_0[indx] = np.asarray(feat_a, dtype=np.float64)
        embeddings_1[indx] = np.asarray(feat_b, dtype=np.float64)

    return embeddings_0, embeddings_1

def load_ijbc_pairs_quality(pair_path):
    with open(pair_path, 'r') as f:
        lines=f.readlines()
    # print(pair_path)
    pairs_quality, targets = [], []

    for idex, line in enumerate(tqdm(lines)):
        # if idex % 1000 == 0:
            # print(idex)
        split_line = line.split()
        pairs_quality.append(float(split_line[3]))  # quality score
        targets.append(int(split_line[2]))   # imposter or genuine
    targets = np.vstack(targets).reshape(-1, )

    # print('Loaded quality score and target for each pair')
    # print('targets.shape: ', targets.shape)
    # print('pairs_quality.shape: ', np.array(pairs_quality).shape)

    return targets, np.array(pairs_quality)

def calculate_pAUC(fnmrs_lists, method_labels, model, output_dir, fmr, db):

    unconsidered_rates = 100 * np.arange(0, 0.32, 0.05)

    for i in range(len(fnmrs_lists)):

        fnmrs_lists_3 = fnmrs_lists[i][:len(unconsidered_rates)]
        fnmrs_list_base = fnmrs_lists[i][0:1]*len(unconsidered_rates)
        # print(fnmrs_list_base)

        auc_value =  metrics.auc(np.array(unconsidered_rates/100), np.array(fnmrs_lists_3))
        auc_value_base = metrics.auc(np.array(unconsidered_rates/100), np.array(fnmrs_list_base))

        print(db, model, method_labels,'pAUC: %.2f' % (auc_value*1000),  'Normalized pAUC: %.3f' %(auc_value/auc_value_base))



def perform_1v1_quality_eval(args):
 d = ['IJBC']
 d = args.eval_db.split(',')


 match = True
 models=args.models.split(',')
 for model in models:
  for dataset in d:

    fnmrs_list_3, method_labels = [], []
    method_labels=[]
    method_names = args.method_name.split(',')

    if (not os.path.isdir(os.path.join(args.output_dir, dataset, 'fnmr'))):
        os.makedirs(os.path.join(args.output_dir, dataset, 'fnmr'))

    # 1. load all features based on number of images
    all_features = load_all_features(root=os.path.join(args.embeddings_dir, f"{dataset}_{model}"))
    # print(all_features.keys())
    unconsidered_rates = np.arange(0, 0.32, 0.05)
    desc = True
    for method_name in method_names:
        print(f"----process {model} {dataset} {method_name}-----------")
        targets, qlts = load_ijbc_pairs_quality(os.path.join(args.quality_score_dir, f"{dataset}_pair_{method_name}.txt"))

        if method_name == 'PFE':
            desc = False 

        if (desc):
            qlts_sorted_idx = np.argsort(qlts)  # [::-1]
        else:
            qlts_sorted_idx = np.argsort(qlts)[::-1]

        num_pairs = len(targets)
        fnmrs_list_3_inner = []

        with open(os.path.join(args.quality_score_dir, f"{dataset}_pair_{method_name}.txt"), 'r') as f:
            lines=f.readlines()
        print('len(lines): ', len(lines))

        hq_pairs_idx = qlts_sorted_idx
        print('hq_pairs_idx.shape: ', hq_pairs_idx.shape)
        print('Calculate distance....')

        dist_pairs = None
        x = []
        y = []
        for indx in range(hq_pairs_idx.shape[0]):
        # for indx in tqdm(range(hq_pairs_idx.shape[0])):
            real_index = hq_pairs_idx[indx]
            split_line = lines[real_index].split()

            feat_a = all_features[(split_line[0] + '.npy')]
            feat_b = all_features[(split_line[1] + '.npy')]

            x.append(np.asarray(feat_a, dtype=np.float64)) 
            y.append(np.asarray(feat_b, dtype=np.float64)) 


            if (indx+1)%10000 == 0:
                x = np.vstack(x)
                y = np.vstack(y)
                dot = np.sum(np.multiply(x, y), axis=1)
                norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
                similarity = np.clip(dot/norm, -1., 1.)
                dist = np.arccos(similarity) / math.pi
                # print(dist)
                if dist_pairs is None:
                    dist_pairs = dist
                else:
                    dist_pairs = np.concatenate((dist_pairs, dist), axis=0)

                del dot, norm, similarity, x, y
                gc.collect()
                x = []
                y = []
            elif indx == (hq_pairs_idx.shape[0]-1):
                x = np.vstack(x)
                y = np.vstack(y)
                dot = np.sum(np.multiply(x, y), axis=1)
                norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
                similarity = np.clip(dot/norm, -1., 1.)
                dist = np.arccos(similarity) / math.pi
                dist_pairs = np.concatenate((dist_pairs, dist), axis=0)
                print(dist_pairs.shape)
                del dot, norm, similarity, x, y
                gc.collect()

        for u_rate in unconsidered_rates:
        # for u_rate in tqdm(unconsidered_rates):
            dist = dist_pairs[int(u_rate * num_pairs):]

            # compute the used paris based on unconsidered rates
            hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs):]

            # load features based on hq_pairs_idx
            # x, y = load_ijbc_pairs_features(os.path.join(args.quality_score_dir, f"{dataset}_pair_{method_name}.txt"), all_features, hq_pairs_idx, args.feat_size)
            # print('Calculate distance....')
            # dist_pairs = []
            # x = []
            # y = []
            # for indx in tqdm(range(hq_pairs_idx.shape[0])):
            #     real_index = hq_pairs_idx[indx]
            #     split_line = lines[real_index].split()
            #     feat_a = all_features[(split_line[0] + '.npy')]
            #     feat_b = all_features[(split_line[1] + '.npy')]
            #     x.append(np.asarray(feat_a, dtype=np.float64)) 
            #     y.append(np.asarray(feat_b, dtype=np.float64)) 
            #     if (indx+1)%100 == 0:
            #         x = np.vstack(x)
            #         y = np.vstack(y)
            #         dot = np.sum(np.multiply(x, y), axis=1)
            #         norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
            #         similarity = np.clip(dot/norm, -1., 1.)
            #         dist = np.arccos(similarity) / math.pi
            #         print(dist)
            #         dist_pairs.append(dist.squeeze())
            #         del dot, norm, similarity, x, y
            #         gc.collect()
            #         x = []
            #         y = []
            # dist_pairs = np.array(dist_pairs)

            # print('Calculate distance....')
            # # if args.distance_metric == 'cosine':
            #     dot = np.sum(np.multiply(x, y), axis=1)
            #     norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
            #     similarity = np.clip(dot/norm, -1., 1.)
            #     dist = np.arccos(similarity) / math.pi
            #     del dot, norm, similarity, x, y
            #     gc.collect()
            # else:
            #     x = sklearn.preprocessing.normalize(x)
            #     y = sklearn.preprocessing.normalize(y)
            #     diff = np.subtract(x, y)
            #     dist = np.sum(np.square(diff), 1)
            #     del diff, x, y
            #     gc.collect()

            # sort in a desending order
            pos_dists =np.sort(dist[targets[hq_pairs_idx] == 1])
            neg_dists = np.sort(dist[targets[hq_pairs_idx] == 0])
            # print('Compute threshold......')
            fmr1000_th = get_eer_threshold(pos_dists, neg_dists, ds_scores=True)
            # fmr100_th, fmr1000_th, fmr10000_th = get_eer_threshold(pos_dists, neg_dists, ds_scores=True)

            # g_true = [g for g in pos_dists if g < fmr100_th]
            # fnmrs_list_2_inner.append(1- len(g_true)/(len(pos_dists)))
            g_true = [g for g in pos_dists if g < fmr1000_th]
            fnmrs_list_3_inner.append(1 - len(g_true) / (len(pos_dists)))
            # g_true = [g for g in pos_dists if g < fmr10000_th]
            # fnmrs_list_4_inner.append(1 - len(g_true) / (len(pos_dists)))
            del pos_dists, neg_dists
            gc.collect()

        # np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr2.npy"), fnmrs_list_2_inner)
        # np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr3.npy"), fnmrs_list_3_inner)
        # np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr4.npy"), fnmrs_list_4_inner)

        # fnmrs_list_2.append(fnmrs_list_2_inner)
        fnmrs_list_3.append(fnmrs_list_3_inner)
        # fnmrs_list_4.append(fnmrs_list_4_inner)
        method_labels.append(f"{method_name}")

    calculate_pAUC(fnmrs_list_3, method_labels, model=model, output_dir=args.output_dir, fmr=1e-3, db=dataset)

def main():
    args = parser.parse_args()
    perform_1v1_quality_eval(args)

if __name__ == '__main__':
    main()
