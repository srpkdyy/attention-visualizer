import torch
import torch.nn as nn
from torchvision import models
from ._visualizer import Visualizer



class BP(Visualizer):
    def __init__(self):
        super().__init__()
        self.backbone = models.vgg16(pretrained=True)
        self.backbone.to(self.device).train()


    def _run(self, img):
        img = img.to(self.device).unsqueeze(0)
        img.requires_grad_()

        out = self.backbone(img)
        idx = out.argmax(1)

        self.backbone.zero_grad()
        bp = torch.eye(1000)[idx].to(self.device, torch.float32)
        out.backward(gradient=bp)

        scores = img.grad.abs().sum(0)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores.cpu()

