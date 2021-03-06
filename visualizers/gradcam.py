import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from torchvision.models import resnet18
from ._visualizer import Visualizer



class GradCAM(Visualizer):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.to(self.device).eval()
        self.untf = TF.Compose([
            TF.ToPILImage(),
            TF.Resize(self.imsize),
            ])

    
    def save_grad(self, grad):
        self.grad = grad

    def _run(self, img):
        x = img.to(self.device).unsqueeze(0)

        for name, module in self.backbone._modules.items():
            x = module(x)
            if name == 'layer4':
                x.register_hook(self.save_grad)
                feature_map = x
            elif name == 'avgpool':
                x = torch.flatten(x, 1)


        idx = x.argmax(1)

        self.backbone.zero_grad()
        bp = torch.eye(1000)[idx].to(self.device, torch.float32)
        x.backward(gradient=bp)

        weights = self.grad.mean(axis=(2, 3), keepdims=True)
        cam = torch.sum((weights * feature_map).squeeze(0), axis=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.cpu()

