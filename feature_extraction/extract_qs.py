import argparse
import os
import sys
import scipy.io as scio
import numpy as np
from tqdm import tqdm
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='lfw',
                        help='test dataset names')
    parser.add_argument('--modelname', type=str, default='ElasticFaceModel',
                        help='ArcFaceModel, CurricularFaceModel, ElasticFaceModel, MagFaceModel')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="../pretrained/ElasticFace-Arc.pth",
                        help='path to pretrained model.')
    parser.add_argument('--model_id', type=str, default="295672",
                        help='digit number in backbone file name')
    parser.add_argument('--relative_dir', type=str, default='../data/quality_data',
                        help='path to save the embeddings')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')
    parser.add_argument('--block', type=str, default="block4", choices=['block1','block2','block3','block4'],
                        help='Specify the neural network layer to apply feature perturbation')
    parser.add_argument('--drop_p', type=float, default=0.5,
                        help='Drop ratio of non-zero elements in the feature map')
    return parser.parse_args(argv)


def read_img_path_list(image_path_file, relative_dir):
    with open(image_path_file, "r") as f:
        lines = f.readlines()
        absolute_list = [line.rstrip() for line in lines]
        lines = [os.path.join(relative_dir, line.rstrip()) for line in lines]
    return lines, absolute_list


def main(param):
    modelname = param.modelname
    gpu_id = param.gpu_id
    data_path = param.relative_dir
    dataset_name = param.dataset_name.split(',')
    block = param.block
    drop_p = param.drop_p


    if modelname == "ArcFaceModel":
        from model.ArcFaceModel import ArcFaceModel
        face_model = ArcFaceModel(param.model_path, gpu_id=gpu_id)
    elif modelname == "ElasticFaceModel":
        from model.ElasticFaceModel import ElasticFaceModel
        face_model = ElasticFaceModel(param.model_path, gpu_id=gpu_id)
    else:
        print("Unknown model")
        exit()


    for d_name in dataset_name:
        print(d_name)
        image_path_list, absolute_list = read_img_path_list(os.path.join(data_path, d_name, "image_path_list.txt"), data_path)
        total_norm, total_pro = face_model.get_quality_scores(image_path_list, block, drop_p)

        print('image nums: ', len(image_path_list))
        print("save quality")


        with open(os.path.join('../data/quality_score', f"{d_name}_{modelname}_PMF.txt"),"w") as f:
            for i in range(len(absolute_list)):
                f.write(absolute_list[i].rstrip()+ " "+str(float(total_pro[i]))+ "\n")




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
