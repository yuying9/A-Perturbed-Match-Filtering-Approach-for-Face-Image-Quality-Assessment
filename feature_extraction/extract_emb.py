import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='../data/quality_data/lfw',
                        help='dataset directory')
    parser.add_argument('--modelname', type=str, default='ArcFaceModel',
                        help='ArcFaceModel, CurricularFaceModel, ElasticFaceModel, MagFaceModel, AdaFaceModel')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="../pretrained/ms1mv3-r100-arcface.pth",
                        help='path to pretrained model.')
    parser.add_argument('--relative_dir', type=str, default='../data/quality_data',
                        help='path to save the embeddings')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')
    return parser.parse_args(argv)



def read_img_path_list(image_path_file, relative_dir):
    with open(image_path_file, "r") as f:
        lines = f.readlines()
        lines = [os.path.join(relative_dir, line.rstrip()) for line in lines]
    return lines

def main(param):
    modelname = param.modelname
    gpu_id = param.gpu_id
    data_path = param.relative_dir
    dataset_path = param.dataset_path

    if modelname == "ArcFaceModel":
        from model.ArcFaceModel import ArcFaceModel
        face_model = ArcFaceModel(param.model_path,  gpu_id=gpu_id)
    elif modelname == "ElasticFaceModel":
        from model.ElasticFaceModel import ElasticFaceModel
        face_model = ElasticFaceModel(param.model_path, gpu_id=gpu_id)
    else:
        print("Unknown model")
        exit()

    dataset_name = dataset_path.split("/")[-1]

    out_path = os.path.join("../data/quality_embeddings", f"{dataset_name}_{modelname}")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(out_path)

    image_path_list = read_img_path_list(os.path.join('../data/quality_data/', dataset_name, "image_path_list.txt"), data_path)

    image_path_list = image_path_list
    print(len(image_path_list))
    
    features = face_model.get_batch_feature(image_path_list)
    features_flipped = face_model.get_batch_feature(image_path_list, flip=1,color=param.color_channel)

    # too slow for IJBC
    # conc_features = np.concatenate((features, features_flipped), axis=1)
    # print(conc_features.shape)

    print("save features")
    for i in tqdm(range(len(features))):
        conc_features = np.concatenate((features[i], features_flipped[i]), axis=0)
        filename = str(str(image_path_list[i]).split("/")[-1].split(".")[0])
        # print(os.path.join(out_path, filename))
        np.save(os.path.join(out_path, filename), conc_features)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
