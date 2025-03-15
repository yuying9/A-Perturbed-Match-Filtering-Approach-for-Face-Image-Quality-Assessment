import os

from backbones.iresnet import iresnet100
from model.FaceModel import FaceModel
import torch
class ArcFaceModel(FaceModel):
    def __init__(self, model_path, gpu_id):
        super(ArcFaceModel, self).__init__(model_path, gpu_id)

    def _get_model(self, ctx, image_size, model_path):
        weight = torch.load(self.model_path)
        backbone = iresnet100().to(f"cuda:{ctx}")
        backbone.load_state_dict(weight, strict=True)
        model = torch.nn.DataParallel(backbone, device_ids=[ctx])
        model.eval()
        return model


    def _getFeatureBlob(self,input_blob,grad=False,mask=None,block=None):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        if grad:
            feat, features, grad = self.model(imgs, grad, mask, block)
            return feat, features, grad
        else:
            feat = self.model(imgs, grad, mask, block)
            return feat