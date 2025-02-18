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
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-mse").to(device)



    transform=transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_prefix="/home/tongji209/majiawei/Dit/dataset/train/source/eroded_"
    gt_prefix="/home/tongji209/majiawei/Dit/dataset/train/target/target_"

    # train_prefix="/home/tongji209/majiawei/Dit/dataset/train/target/target_"
    # gt_prefix="/home/tongji209/majiawei/Dit/dataset/train/source/eroded_"

    # y0=Image.open("/home/tongji209/majiawei/Dit/dataset/train/source/eroded_0.jpg").convert("RGB")
    # y0=transform(y0)
    # y0=y0.unsqueeze(0)

    # y1=Image.open("/home/tongji209/majiawei/Dit/dataset/train/source/eroded_1.jpg").convert("RGB")
    # y1=transform(y1)
    # y1=y1.unsqueeze(0)

    # y2=Image.open("/home/tongji209/majiawei/Dit/dataset/train/source/eroded_2.jpg").convert("RGB")
    # y2=transform(y2)
    # y2=y2.unsqueeze(0)

    # y3=Image.open("/home/tongji209/majiawei/Dit/dataset/train/source/eroded_3.jpg").convert("RGB")
    # y3=transform(y3)
    # y3=y3.unsqueeze(0)

    # y=torch.cat((y0,y1,y2,y3),0).to(device)
    # sample_idx=[0,1,2,3,4,5,6,7,8,9]
    sample_idx=[0]
    img_list=[]
    for i in sample_idx:
        yy=Image.open(train_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        img_list.append(yy)

    

    y=torch.cat(tuple(img_list),0).to(device)


    # save_image(y, "sample-inapint-2.png", nrow=4, normalize=True, value_range=(-1, 1))

    # return

    sample_num=y.shape[0]

    # Create sampling noise:
    # n = len(class_labels)
    z = torch.randn(sample_num, 4, latent_size, latent_size, device=device)

    # y = torch.tensor(class_labels, device=device)




    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y_null = torch.tensor([args.num_classes] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    


    # print(y.shape)

    # return 
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    model_kwargs = dict(y=y)
    

    # # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    gt_list=[]
    for i in sample_idx:
        yy=Image.open(gt_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        gt_list.append(yy)

    ggtt=torch.cat(tuple(gt_list),0).to(device)

    y_enc = vae.encode(y).latent_dist.sample().mul_(0.18215)
    ggtt_enc=vae.encode(ggtt).latent_dist.sample().mul_(0.18215)
    print(mean_flat((ggtt_enc - y_enc) ** 2))
    # sample_loss=mean_flat((ggtt - samples) ** 2)
    # print(sample_loss.shape)
    # print(sample_loss)

        
    if args.result_dir:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir,exist_ok=True)
        img_index = len(glob(f"{args.result_dir}/*"))
        img_folder_path=os.path.join(args.result_dir,f"sample-inapint-{img_index:03d}")
        
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path,exist_ok=True)
        

    noise = torch.randn_like(ggtt_enc)
    t = torch.randint(0, diffusion.num_timesteps, (ggtt_enc.shape[0],), device=device)
    x_t = diffusion.q_sample(ggtt_enc, t, noise=noise)
    print(mean_flat((ggtt_enc - x_t) ** 2))

    x_t_dec=vae.decode(x_t / 0.18215).sample

    save_image(x_t_dec, os.path.join(img_folder_path,"x_t_dec.png"), nrow=4, normalize=True, value_range=(-1, 1))

    return 
    sample_cnt=0
    # 可视化
    for sample in diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    ):
        
        # sample = vae.decode(sample / 0.18215).sample
        # with torch.no_grad():
        #     samples = vae.decode(samples / 0.18215).sample
        #     canvas=generate_img_canvas(samples,args.image_size)
        #     canvas_path=os.path.join("/public/home/acr0vd9ik6/project/DiT/fast-DiT/sample_result/sample-visualize","canvas-"+f"{cnt:04d}.png")
        #     canvas.save(canvas_path)
        sample_loss=mean_flat((ggtt_enc - sample) ** 2)
        sample = vae.decode(sample / 0.18215).sample
        if sample_cnt% args.sample_visual_every == 0:
            visual_img_path=os.path.join(img_folder_path,"canvas-"+f"{sample_cnt:04d}.png")
            save_image(sample, visual_img_path, nrow=4, normalize=True, value_range=(-1, 1))
        
        with open(os.path.join(img_folder_path,"loss.txt"),"a") as f:
            f.write(f"step-{sample_cnt}-loss : ")
            for loss in sample_loss:
                f.write(f"{loss:.6f} " )
            f.write("\n")
        sample_cnt+=1
        # print(sample_cnt)


        # t = torch.tensor([args.num_sampling_steps-1] * sample_num, device=device)
        # with torch.no_grad():
        #     z_decode=vae.decode(z / 0.18215).sample
        #     out = diffusion.p_sample(
        #         model,
        #         z,
        #         t,
        #         clip_denoised=False,
        #         model_kwargs=model_kwargs,
        #     )
        #     out_sample=vae.decode(out["sample"] / 0.18215).sample
        #     out_pred=vae.decode(out["pred_xstart"] / 0.18215).sample
        #     sample=torch.cat((z_decode,out_sample,out_pred),0)
        #     # sample = vae.decode(sample / 0.18215).sample


    if args.result_dir=="":
        # Save and display images:
        # save_image(samples, "sample-inapint.png", nrow=4, normalize=True, value_range=(-1, 1))
        pass
    else:
        # if not os.path.exists(args.result_dir):
            # os.makedirs(args.result_dir,exist_ok=True)
        # img_index = len(glob(f"{args.result_dir}/*"))
        # img_folder_path=os.path.join(args.result_dir,f"sample-inapint-{img_index:03d}")
        
        # img_name=f"sample-inapint-{img_index:03d}.png"
        # if not os.path.exists(img_folder_path):
            # os.makedirs(img_folder_path,exist_ok=True)
        
        img_name=f"generated.png"
        # save_image(sample, os.path.join(img_folder_path,img_name), nrow=4, normalize=True, value_range=(-1, 1))
        save_image(sample, os.path.join(img_folder_path,img_name), nrow=4, normalize=True, value_range=(-1, 1))
        save_gt_img(transform,img_folder_path,gt_prefix,sample_idx)
        save_train_img(transform,img_folder_path,train_prefix,sample_idx)
        # with open(os.path.join(img_folder_path,"loss.txt"),"w") as f:
        #     for loss in sample_loss:
        #         f.write(f"loss : {loss:.6f} \n" )
            





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
    args = parser.parse_args()
    main(args)
