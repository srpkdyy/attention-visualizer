import torch
from torchvision import transforms as TF
from torchvision.models import resnet18
from ._visualizer import Visualizer


class CAM(Visualizer):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.to(self.device).eval()
        self.untf = TF.Compose([
            TF.ToPILImage(),
            TF.Resize(self.imsize),
            ])


    def _run(self, img):
        x = img.to(self.device).unsqueeze(0)

        for name, module in self.backbone._modules.items():
            if name == 'avgpool':
                feature_map = x
                x = module(x)
                x = torch.nn.Flatten()(x)
            else :
                x = module(x)

        idx = x.argmax(1)
        weights = self.backbone.fc.weight[idx]
        weights = weights.reshape(1, -1, 1, 1)

        cam = torch.sum((weights * feature_map).squeeze(0), axis=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.cpu()

