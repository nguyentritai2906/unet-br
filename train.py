import argparse
import copy
import logging
import os

import numpy as np
import torch
import torchvision
from torch.nn.parallel import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.functional import accuracy, f1, psnr

from model.criterion import HeScho
from model.dataset import DIBCO
from model.model import UNetBR
from model.transform import (Compose, Grayscale, RandomCrop, RandomEqualize,
                             RandomInvert, RandomRotation, RandomScale,
                             ToTensor)
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--data_dir',
                        help='data directory',
                        type=str,
                        default='./dataset/DIBCO')
    parser.add_argument('--load',
                        help='path to load weights from',
                        type=str,
                        default='./weight/weights.pth')
    parser.add_argument('--save',
                        help='path to save weights',
                        type=str,
                        default='./weight/weights.pth')
    parser.add_argument('--num_blocks',
                        help='number of UNet blocks',
                        type=int,
                        default=2)
    parser.add_argument('--batch_size', help='batch size', type=int, default=8)
    parser.add_argument('--crop_size', help='crop size', type=int, default=256)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--num_epochs',
                        help='number of epochs',
                        type=int,
                        default=100)
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    datadir = args.data_dir
    load = args.load
    save = args.save
    crop = args.crop_size
    n_epochs = args.num_epochs
    lr = args.lr
    num_blocks = args.num_blocks
    batch_size = args.batch_size

    logger = logging.getLogger('Train')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger("./log/log.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 42
    validation_split = .2
    shuffle_dataset = True

    load_from = load
    weights_path = save

    scale_train = [.75, 1.5, .25]
    scale_val = [1, 1, 0]
    lr_step_size = 50

    data_transforms = {
        'train':
        Compose([
            Grayscale(),
            RandomInvert(1),
            RandomScale(*scale_train),
            RandomCrop(crop, crop,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       True),
            RandomRotation(270),
            ToTensor(),
        ]),
        'val':
        Compose([
            Grayscale(),
            RandomInvert(1),
            RandomScale(*scale_val),
            RandomCrop(crop, crop,
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       tuple([int(v * 255) for v in [0.485, 0.456, 0.406]]),
                       False),
            ToTensor(),
        ]),
    }

    datasets = {
        x: DIBCO(datadir, transform=data_transforms[x])
        for x in ('train', 'val')
    }

    model = UNetBR(num_blocks)
    model = DataParallel(model)

    try:
        checkpoint = torch.load(load_from)
        weights = checkpoint['model_state_dict']
        model.load_state_dict(weights, strict=False)
        best_f1 = checkpoint['best_f1']
        print(f"Best F1 Score: {best_f1:.4f}")
    except:
        best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)

    criterion = HeScho()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=lr_step_size,
                                    gamma=0.1)

    # Dataloader
    dataset_size = len(datasets['train'])
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(datasets['train'],
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=2)
    validation_loader = DataLoader(datasets['val'],
                                   batch_size=batch_size,
                                   sampler=valid_sampler,
                                   num_workers=2)
    dataloaders = {'train': train_loader, 'val': validation_loader}

    # Training loop
    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            f1_score = 0.0
            psnr_score = 0.0
            acc_score = 0.0

            loader = dataloaders[phase]
            n_total_steps = len(loader)

            for _, (img, gt, fnames) in enumerate(loader):
                img = img.to(device)
                gt = gt.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img)
                    loss = criterion(output, gt)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * img.size(0)
                f1_score += f1(output[-1], gt.data.type(torch.uint8))
                acc_score += accuracy(output[-1], gt.data.type(torch.uint8))
                psnr_score += psnr(output[-1], gt.data)

            if phase == 'train':
                scheduler.step()

            if (epoch + 1) % 1 == 0:
                out = output[-1][0]
                gt = gt[0]
                img = img[0]
                fname = fnames[0].split('/')[-1]
                fname = os.path.join("./log/debug_imgs", fname)
                try:
                    torchvision.utils.save_image(torch.stack((img, out, gt)),
                                                 fname)
                except:
                    os.makedirs("./log/debug_imgs")
                    torchvision.utils.save_image(torch.stack((img, out, gt)),
                                                 fname)

            epoch_loss = running_loss / n_total_steps
            epoch_acc = acc_score / n_total_steps
            epoch_f1 = f1_score / n_total_steps
            epoch_psnr = psnr_score / n_total_steps

            msg = "Epoch: {}/{} \t {} \t Loss: {:.4f} F1: {:.4f} Acc: {:.4f}, PSNR: {:.4f}".format(
                epoch + 1, n_epochs, "Train" if phase == 'train' else "Val",
                epoch_loss, epoch_f1, epoch_acc, epoch_psnr)
            logger.info(msg)

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'best_f1': best_f1
                    }, weights_path)
                best_model_wts = copy.deepcopy(model.state_dict())
                msg = "Saved best model. F1 Score: {:.4f}".format(epoch_f1)
                logger.info(msg)

    print('Best val F1 Score: {:.4f}'.format(best_f1))
    torch.save({
        'model_state_dict': best_model_wts,
        'best_f1': best_f1
    }, weights_path)


if __name__ == "__main__":
    main()
