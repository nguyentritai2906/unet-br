import argparse
import os

import torch
import torchvision
from PIL import Image
from torch.nn.parallel import DataParallel
from torchvision import transforms as T
from tqdm import tqdm

from model.model import UNetBR
from utils.util import crop


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('files',
                        metavar='F',
                        type=str,
                        nargs='+',
                        help='Images to remove background')
    parser.add_argument('--load',
                        help='path to load weights from',
                        type=str,
                        default='./weight/weights.pth')
    parser.add_argument('--num_block',
                        help='number of UNet blocks',
                        type=int,
                        default=2)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    files = args.files
    load = args.load
    num_block = args.num_block

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transforms = T.Compose([
        T.RandomInvert(1),
        T.Grayscale(),
        T.ToTensor(),
    ])

    model = UNetBR(num_block)
    model = DataParallel(model)
    checkpoint = torch.load(load, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        for fname in tqdm(files, total=len(files)):
            print(fname)
            img = Image.open(fname)
            img = data_transforms(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)
            output = model(img)
            out = output[-1][0]
            fname = os.path.join(*fname.split('/')[:-1],
                                 "processed-" + os.path.basename(fname))
            torchvision.utils.save_image(T.RandomInvert(1)(out), fname)


if __name__ == "__main__":
    main()
