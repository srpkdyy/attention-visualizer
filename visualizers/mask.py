import torch
from torchvision import models
from ._visualizer import Visualizer



class ZeilerFergus(Visualizer):
    def __init__(self, ksize=64, stride=8):
        super().__init__()
        self.ksize = 64
        self.stride = 8
        self.n_masking = (self.imsize + ksize) // stride - 1
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.to(self.device).eval()


    def run(self, imgs):
        imgs = self.transform(imgs)
        outputs = [self._run(img) for img in imgs]
        outputs = self.untransform(outputs)
        return outputs


    def _run(self, img):
        masked = self._mask(img)
        img, masked = img.to(self.device), masked.to(self.device)

        with torch.no_grad():
            out = self.backbone(img.unsqueeze(0)).squeeze(0)
            conf, class_idx = out.max(), out.argmax()

            out = self.backbone(masked)
            scores = out[:, class_idx]


        scores = scores.detach().cpu()
        heatmap = self._reconstruct(scores)
        return heatmap


    def _mask(self, img):
        imsize = self.imsize
        ksize = self.ksize
        stride = self.stride

        out = img.repeat(self.n_masking ** 2, 1, 1, 1)

        base = stride - ksize
        for i in range(self.n_masking):
            for j in range(self.n_masking):
                idx = self.n_masking * i + j
                ymin = base + stride * i
                ymax = ymin + ksize
                xmin = base + stride * j
                xmax = xmin + ksize

                out[idx, :, max(ymin, 0):min(ymax, imsize), max(xmin, 0):min(xmax, imsize)] = 0
        return out

    
    def _reconstruct(self, diff):
        imsize = self.imsize
        ksize = self.ksize
        stride = self.stride

        out = torch.zeros((self.imsize, self.imsize))

        base = stride - ksize
        for i in range(self.n_masking):
            for j in range(self.n_masking):
                idx = self.n_masking * i + j
                ymin = base + stride * i
                ymax = ymin + ksize
                xmin = base + stride * j
                xmax = xmin + ksize

                out[max(ymin, 0):min(ymax, imsize), max(xmin, 0):min(xmax, imsize)] += diff[idx]
        return (out - out.min()) / (out.max() - out.min())

