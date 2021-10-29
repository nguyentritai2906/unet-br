import numpy as np
import torch
import torchvision
from PIL import Image


def OTSU(im_gray):
    img_size = im_gray.shape[-1]**2
    g_max = 0
    best_th = 0
    best_u0 = 0.0
    G = np.zeros(100)

    for threshold in np.arange(0, 1, 0.01):
        logic0 = im_gray > threshold
        logic1 = im_gray <= threshold

        fore_pix = torch.sum(logic0)
        back_pix = torch.sum(logic1)
        if fore_pix == 0:  #END
            break
        if back_pix == 0:  #CONTINUE
            continue

        w0 = float(fore_pix) / img_size
        w1 = float(back_pix) / img_size

        u0 = float(torch.sum(im_gray * logic0)) / fore_pix
        u1 = float(torch.sum(im_gray * logic1)) / back_pix

        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        G[int(threshold * 100)] = g
        if g_max < g:
            g_max = g
            best_th = threshold
            best_u0 = u0
            best_u1 = u1
    return best_th


def crop(img, fname):
    """
    Crop image to it's original ratio
    """
    cur_size = img.shape[-1]
    target_size = Image.open(fname).size
    ratio = float(cur_size) / max(target_size)
    W, H = tuple([int(x * ratio) for x in target_size])
    x1, y1 = (cur_size - W) // 2, (cur_size - H) // 2
    img = torchvision.transforms.ToPILImage()(img)
    return img.crop((x1, y1, x1 + W, y1 + H))


def save_ori_out(outs, fnames):
    for out, fname in zip(outs, fnames):
        out = crop(out, fname)
        W, H = out.size
        ori = Image.open(fname).resize((W, H))
        img = Image.new("RGB", (W * 2 + 10, H), "white")
        img.paste(ori, (0, 0))
        img.paste(out, (W + 10, 0))
        fname = fname.split('/')
        fname = '/'.join((*fname[:2], "out", *fname[3:]))
        torchvision.utils.save_image(torchvision.transforms.ToTensor()(img),
                                     fname)


def save_out(outs, fnames):
    for out, fname in zip(outs, fnames):
        out = crop(out, fname)
        fname = fname.split('/')
        fname = '/'.join((*fname[:2], "out", *fname[3:]))

        torchvision.utils.save_image(torchvision.transforms.ToTensor()(out),
                                     fname)
