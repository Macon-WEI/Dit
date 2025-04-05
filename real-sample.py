# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

import argparse
import os
from glob import glob
from PIL import Image
import sys
import torchvision.transforms as transforms

sys.path.append(os.path.abspath("./train_options"))

from train_options.models_inpaint import DiT_models
from train_options.train_inpaint import center_crop_arr
import random


def save_gt_img(transform,result_dir,gt_prefix,sample_idx):
    img_list=[]
    for i in sample_idx:
        yy=Image.open(gt_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        img_list.append(yy)

    y=torch.cat(tuple(img_list),0)

    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir,exist_ok=True)
    # img_index = len(glob(f"{result_dir}/*"))
    img_name=f"gt.png"
    save_image(y, os.path.join(result_dir,img_name), nrow=4, normalize=True, value_range=(-1, 1))

def save_train_img(transform,result_dir,train_prefix,sample_idx):
    img_list=[]
    for i in sample_idx:
        yy=Image.open(train_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        img_list.append(yy)

    y=torch.cat(tuple(img_list),0)

    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir,exist_ok=True)
    # img_index = len(glob(f"{result_dir}/*"))
    img_name=f"source.png"
    save_image(y, os.path.join(result_dir,img_name), nrow=4, normalize=True, value_range=(-1, 1))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def main(args):

    # torch.cuda.set_device(6)  # 设置默认 GPU 为 ID 0


    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        # assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes,
        con_img_size=args.image_size,
        con_img_channels=3,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    # diffusion = create_diffusion(str(args.num_sampling_steps),diffusion_steps=args.diffusion_steps,predict_xstart=args.predict_xstart)
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=args.diffusion_steps,predict_xstart=args.predict_xstart)
    
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-mse").to(device)



    transform=transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # train_prefix="/remote-home/zhangxinyue/DiT/train/source/eroded_"
    # gt_prefix="/remote-home/zhangxinyue/DiT/train/target/target_"

    # train_prefix="/remote-home/zhangxinyue/DiT/test/task1/eroded/test1_"
    # gt_prefix="/remote-home/zhangxinyue/DiT/test/task1/target/target1_"

    train_prefix="/home/tongji209/majiawei/Dit/dataset/train/target/target_"
    gt_prefix="/home/tongji209/majiawei/Dit/dataset/train/source/eroded_"
    unrestored_images=os.listdir(args.test_path)
    unrestored_image_num=len(os.listdir(args.test_path))

    if args.result_dir:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir,exist_ok=True)
        
        folder_index = len(glob(f"{args.result_dir}/*"))
        img_folder_path=os.path.join(args.result_dir,f"sample-{folder_index:03d}")
        
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path,exist_ok=True)
    result_prefix="sample"
    print(unrestored_images)

    for img in unrestored_images:
        img_extend_name=img.split(".")[-1]
        img_idx=img.split(".")[0].split("_")[1]
        image=Image.open(os.path.join(args.test_path,img)).convert("RGB")
        image_trans=transform(image)
        image_trans=image_trans.unsqueeze(0).to(device)
        model_kwargs = dict(y=image_trans)

        image_enc=vae.encode(image_trans).latent_dist.sample().mul_(0.18215)


        # for sample in diffusion.p_sample_loop(
        #     model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        # ):
        
        sample=diffusion.p_sample_loop(
            model.forward, image_enc.shape, image_enc, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        sample = vae.decode(sample / 0.18215).sample
        visual_img_path=os.path.join(img_folder_path,result_prefix+"_"+img_idx+"."+img_extend_name)
        save_image(sample, visual_img_path, nrow=1, normalize=True, value_range=(-1, 1))
        # input("waiting")
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--test-path", type=str, required=True)
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--result-dir", type=str, default="")
    parser.add_argument("--sample-visual-every", type=int, default=10)
    parser.add_argument("--predict-xstart", action='store_true', help="use predict xstart in diffusion training")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    args = parser.parse_args()
    main(args)
