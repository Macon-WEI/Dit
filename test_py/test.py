import sys
sys.path.append("/public/home/acr0vd9ik6/project/DiT/fast-DiT/")
sys.path.append("/public/home/acr0vd9ik6/project/DiT/fast-DiT/train_options/")
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os


from models_original import DiT_models,LabelEmbedder

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def load_image(img_path,image_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    img=Image.open(img_path)
    img_trs=transform(img)
    img_batch=img_trs.unsqueeze(0)

    return img_batch




def main():
    assert torch.cuda.is_available,"torch.cuda.is_available is false"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device=
    print("this is device",device)
    vae = AutoencoderKL.from_pretrained(f"/public/home/acr0vd9ik6/project/DiT/fast-DiT/sd-vae-ft-mse").to(device)
    img_path="/public/home/acr0vd9ik6/project/DiT/fast-DiT/data1/final/final/img_0.jpg"
    img_size=256
    img_=load_image(img_path,img_size)
    img_.to(device)
    with torch.no_grad():
        x=vae.encode(img_).latent_dist.sample().mul_(0.18215)
    
    print(x.shape)
    num_classes=3
    hidden_size=1024
    class_dropout_prob=0.1
    class_labels = [1,2,3]

    # Create sampling noise:
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)
    print("this is y",y,type(y))

    # y_null = torch.tensor([1000] * n, device=device)
    y_null = torch.tensor([3] * n, device=device)
    print("this is y_null",y_null,type(y_null))
    y = torch.cat([y, y_null], 0)
    print("this is y + y_null",y,type(y))

    label_emb=LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
    y_emb=label_emb(y,False)
    print("this is y_emb",y_emb,type(y_emb))



if __name__=="__main__":
    main()