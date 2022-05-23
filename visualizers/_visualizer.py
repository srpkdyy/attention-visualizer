import torch
from torchvision import transforms as TF
from tqdm import tqdm

class Visualizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.imsize = 256
        self.tf = TF.Compose([
            TF.ToTensor(),
            TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.untf = TF.Compose([
            TF.ToPILImage(),
            ])

    def run(self, imgs):
        imgs = self.transform(imgs)
        outputs = [self._run(img) for img in tqdm(imgs)]
        outputs = self.untransform(outputs)
        return outputs

    def _run(self, img):
        pass

    def transform(self, imgs):
        return [self.tf(img) for img in imgs]

    def untransform(self, outputs):
        return [self.untf(out) for out in outputs]

