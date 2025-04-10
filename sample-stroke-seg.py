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

from train_options.models_stroke_seg import DiT_models
from train_options.train_stroke_seg import center_crop_arr
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



    train_prefix="/home/tongji209/majiawei/stroke_segmentation/train_debug/characters-10/"
    gt_prefix="/home/tongji209/majiawei/stroke_segmentation/train_debug/strokes-10/"

    
    sample_idx=[19971,19972,19975,19976,19978]
    stroke_sample_idx=[2,1,3,3,2]
    # gt_sample_idx=[19971_2,19972_1,19975_3,19976_3,19978_2]
    gt_sample_idx=[]

    for i in range(len(sample_idx)):
        gt_sample_idx.append(str(sample_idx[i])+"_"+str(stroke_sample_idx[i]))
    




    img_list=[]

    

    for i in sample_idx:
        yy=Image.open(train_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        img_list.append(yy)

    cond=torch.cat(tuple(img_list),0).to(device)

    stroke_sample_tensor=torch.tensor(stroke_sample_idx).to(device)

    unicode_sample_tensor=torch.tensor(sample_idx).to(device)


    sample_num=cond.shape[0]

    # Create sampling noise:
    # n = len(class_labels)
    zz = torch.randn(sample_num, 4, latent_size, latent_size, device=device)
    zzz=vae.encode(cond).latent_dist.sample().mul_(0.18215)

    z=torch.cat((zz,zzz),0)

    eval_model_kwargs = dict(y=torch.cat((cond,cond),0),stroke_order=torch.cat((stroke_sample_tensor,stroke_sample_tensor),0),unicode=torch.cat((unicode_sample_tensor,unicode_sample_tensor),0))

    gt_list=[]
    for i in gt_sample_idx:
        yy=Image.open(gt_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        gt_list.append(yy)

    ggtt=torch.cat(tuple(gt_list),0).to(device)

    cond_enc = vae.encode(cond).latent_dist.sample().mul_(0.18215)
    ggtt_enc=vae.encode(ggtt).latent_dist.sample().mul_(0.18215)

    if args.result_dir:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir,exist_ok=True)
        
        img_index = len(glob(f"{args.result_dir}/*"))
        print(f"img_index {img_index}")
        img_folder_path=os.path.join(args.result_dir,f"sample-inapint-{img_index:03d}")
        
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path,exist_ok=True)
        
    
    

    sample_cnt=0
    # print(eval_model_kwargs)
    # 可视化
    for sample in diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=eval_model_kwargs, progress=True, device=device
    ):
        
        # sample = vae.decode(sample / 0.18215).sample
        # with torch.no_grad():
        #     samples = vae.decode(samples / 0.18215).sample
        #     canvas=generate_img_canvas(samples,args.image_size)
        #     canvas_path=os.path.join("/public/home/acr0vd9ik6/project/DiT/fast-DiT/sample_result/sample-visualize","canvas-"+f"{cnt:04d}.png")
        #     canvas.save(canvas_path)
        sample = vae.decode(sample / 0.18215).sample
        if sample_cnt% args.sample_visual_every == 0:
            visual_img_path=os.path.join(img_folder_path,"canvas-"+f"{sample_cnt:04d}.png")
            sample_compare=torch.cat((sample,ggtt),0).to(device)
            save_image(sample_compare, visual_img_path, nrow=len(sample_idx), normalize=True, value_range=(-1, 1))
        # sample_loss=mean_flat((ggtt - sample) ** 2)
        # with open(os.path.join(experiment_dir,"loss.txt"),"a") as f:
        #     f.write(f"step-{sample_cnt}-loss : ")
        #     for loss in sample_loss:
        #         f.write(f"{loss:.6f} " )
        #     f.write("\n")
        sample_cnt+=1

    if args.result_dir=="":
        pass
    else:
        img_name=f"generated.png"
        save_image(sample, os.path.join(img_folder_path,img_name), nrow=len(sample_idx), normalize=True, value_range=(-1, 1))
        save_gt_img(transform,img_folder_path,gt_prefix,gt_sample_idx)
        save_train_img(transform,img_folder_path,train_prefix,sample_idx)
            





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
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
