import cv2
import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn


def get_mask(cam,drop_p=0.5):
    cam_shape = cam.shape
    c = torch.where(cam>0,torch.ones_like(cam),torch.zeros_like(cam))
    c = c.sum().data.cpu().numpy()
    
    mask = torch.ones_like(cam).cuda()
    cam = cam.reshape((cam_shape[0],-1))
    mask = mask.reshape((cam_shape[0],-1))
    _, index = torch.topk(cam, int(c*drop_p), -1)
    mask[0,index[0]] = 0
    mask = mask.reshape(cam_shape)

    return mask.detach()



class FaceModel():
    def __init__(self, model_path, ctx_id):
        self.gpu_id=ctx_id
        self.image_size = (112, 112)
        self.model_path=model_path

        self.model=self._get_model(ctx=ctx_id,image_size=self.image_size,model_path=self.model_path)

    def _get_model(self, ctx, image_size, model_path):
        pass

    def _getFeatureBlob(self,input_blob):
        pass

    def get_feature(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112))
        a = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        emb=self._getFeatureBlob(input_blob)
        emb = normalize(emb.reshape(1, -1))
        return emb


    def get_batch_feature(self, image_path_list, batch_size=64, flip=0, color="BGR"):
        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        for i in range(0, len(image_path_list), batch_size):

            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
            else:
                tmp_list = image_path_list[i :]
            count += 1

            images = []
            for image_path in tmp_list:
                image = cv2.imread(image_path)

                if image is None:
                    continue
                image = cv2.resize(image, (112, 112))
                if (color == "RGB"):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if flip == 1:
                    image = cv2.flip(image, 1)
                a = np.transpose(image, (2, 0, 1))
                images.append(a)
            input_blob = np.array(images)
            with torch.no_grad():
                emb = self._getFeatureBlob(input_blob)
            features.append(emb.data.cpu().numpy())
            print("batch"+str(i))
        features = np.vstack(features)
        features = normalize(features)
        return features



    def get_quality_scores(self, image_path_list, block, drop_p, flip=0, color="BGR"):
        H = 112

        linear = torch.nn.Linear(512,1,bias=False).cuda()

        total_norm = []
        total_pro = []

        x_black = np.zeros((1,3,112,112))
        with torch.no_grad():
            f_black = self._getFeatureBlob(x_black, grad=False)


        for i, image_path in enumerate(image_path_list):
            if (i+1)%100 == 0:
                print(i)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (H, H))
            if (color == "RGB"):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip == 1:
                image = cv2.flip(image, 1)  

            a = np.transpose(image, (2, 0, 1))
            x_ori = np.expand_dims(a, axis=0)

            f_ori, features, grads = self._getFeatureBlob(x_ori, grad=True)

            f_ori_norm = torch.norm(f_ori, p=2)

            alpha = ((f_ori*f_ori).sum(-1) + (f_ori*f_black).sum(-1)) / 2.0
            beta = ((f_ori*f_ori).sum(-1) - (f_ori*f_black).sum(-1)) / 10.0
            
            alpha = alpha.detach()
            beta = beta.detach()

            linear.weight = torch.nn.Parameter(f_ori)
            qs = linear(f_ori)

            ######################################################
            # # The two lines can be commented out; using qs.backward() yields the same mask faster.
            loss = (qs - alpha)/beta
            loss = torch.sigmoid(loss)
            ######################################################

            self.model.zero_grad()
            loss.backward()
            
            g = grads[block]
            f = features[block]

            cam = torch.nn.functional.relu(f*g).detach()

            mask = get_mask(cam, drop_p)

            with torch.no_grad():
                f_mask = self._getFeatureBlob(x_ori, mask=mask, block=block)

            n = torch.norm(f_mask, p=1)
            qs = linear(f_mask)

            p = (qs - alpha) / beta
            p = torch.sigmoid(p)

            total_norm.append(n.data.cpu().numpy())
            total_pro.append(p.data.cpu().numpy())


        total_norm = np.vstack(total_norm)
        total_pro = np.vstack(total_pro)

        return total_norm, total_pro


