import os
import cv2
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms as TF


parser = argparse.ArgumentParser()
parser.add_argument('indir')
parser.add_argument('outdir')
parser.add_argument('-m', '--method', default='mask', help='|mask|bp|cam|')

def overlay_heatmap(img, score, method):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.array(score), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)


def main(args):
    if args.method == 'mask':
        from visualizers.mask import ZeilerFergus
        visualizer = ZeilerFergus()
    elif args.method == 'bp':
        from visualizers.bp import BP
        visualizer = BP()

    preprocess = TF.Compose([TF.Resize(visualizer.imsize), TF.CenterCrop(visualizer.imsize)])

    filenames = os.listdir(args.indir)
    imgs = [preprocess(Image.open(args.indir + fname)) for fname in filenames]

    scores = visualizer.run(imgs)
    heatmaps = map(overlay_heatmap, imgs, scores, [args.method]*len(imgs))

    for hm, fname in zip(heatmaps, filenames):
        cv2.imwrite(args.outdir + '_{}.'.format(args.method).join(fname.split('.')), hm)


if __name__ == '__main__':
    main(parser.parse_args())

