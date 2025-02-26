# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' # 防止内存碎片化，提高内存分配效率，减少内存分配失败


from models_inpaint import DiT_models
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from data.dataset import PairedImageDataset
from visualize import visualize_log
from diffusion.gaussian_diffusion import mean_flat

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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




#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."



    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    device_str=f"cuda:{device}"
    # print("-------device_str--------",device_str)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    print("device",type(device),device)
    # return

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"train_data_path : {args.train_data_path}")
        logger.info(f"gt_data_path : {args.gt_data_path}")
        logger.info(f"global-batch-size : {args.global_batch_size}")
        logger.info(f"learning-rate : {args.learning_rate}")
        logger.info(f"epochs : {args.epochs}")
        logger.info(f"diffusion_steps : {args.diffusion_steps}")
        logger.info(f"predict_xstart : {args.predict_xstart}")

    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes,
        con_img_size=args.image_size,
        con_img_channels=3,
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    if args.resume_from_checkpoint:
        assert os.path.isfile(args.resume_from_checkpoint), f'Could not find DiT checkpoint at {args.resume_from_checkpoint}'
        # state_dict=torch.load(args.resume_from_checkpoint,map_location=lambda storage, loc: storage)
        # state_dict=torch.load(args.resume_from_checkpoint,map_location=device_str)
        state_dict=torch.load(args.resume_from_checkpoint)
        model.load_state_dict(state_dict['model'])
        ema.load_state_dict(state_dict['ema'])
        opt.load_state_dict(state_dict['opt'])
        logger.info(f"Resumed training from checkpoint {args.resume_from_checkpoint} !")

    # print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    # print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)

    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=args.diffusion_steps,predict_xstart=args.predict_xstart)  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-mse").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)

    # train_dir="/home/tongji209/majiawei/Dit/dataset/train/eroded"
    # gt_dir="/home/tongji209/majiawei/Dit/dataset/train/real"
    train_dir=args.train_data_path
    gt_dir=args.gt_data_path
    dataset=PairedImageDataset(train_dir,gt_dir,transform)
    print(train_dir,gt_dir)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    print("batch_size=",int(args.global_batch_size // dist.get_world_size()))
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} train images ({args.train_data_path})")
    logger.info(f"Dataset contains {len(dataset):,} gt images ({args.gt_data_path})")

    # return
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    visual_num=min(10,int(args.global_batch_size // dist.get_world_size()))

    logger.info(f"Training for {args.epochs} epochs...")
    min_loss=100
    best_model_ckpt=None
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        per_epoch_running_loss=0.0
        for x, y in loader:
            # logger.info(f"train_steps {train_steps}")
            x = x.to(device)
            y = y.to(device)
            source_img=x[:visual_num].cpu().clone()
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                y = vae.encode(y).latent_dist.sample().mul_(0.18215)

                # x_enc = vae.encode(x).latent_dist.sample().mul_(0.18215)


            # t = torch.randint(0, diffusion.num_timesteps, (y.shape[0],), device=device)     # 改成100，
            t = torch.randint(diffusion.num_timesteps-1, diffusion.num_timesteps, (y.shape[0],), device=device)     # 改成100，
            # t=torch.full(y.shape[0],99,device=device)
            model_kwargs = dict(y=x)
            loss_dict = diffusion.training_losses(model, y, t, model_kwargs)
            # loss_dict = diffusion.training_losses(model, x_enc, t, model_kwargs,clean_target=y)



            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            per_epoch_running_loss+=loss.item()
            log_steps += 1
            train_steps += 1
            # print("train_steps",train_steps)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                # print("loss_dict['model_output'].shape ",loss_dict['model_output'].shape)
                # print("loss_dict['target'].shape ",loss_dict['target'].shape)

            if train_steps % args.visual_every == 0:
                with torch.no_grad():
                    x_t_detach=vae.decode(loss_dict['x_t'][:visual_num]/ 0.18215).sample.to("cpu")
                    model_output_detach=vae.decode(loss_dict['model_output'][:visual_num]/ 0.18215).sample.to("cpu")
                    target_detach=vae.decode(loss_dict['target'][:visual_num]/ 0.18215).sample.to("cpu")
                    pre_dict=vae.decode(loss_dict['pred_xstart'][:visual_num]/ 0.18215).sample.to("cpu")
                    visual_img_tensor=torch.cat((source_img,x_t_detach,model_output_detach,pre_dict,target_detach),0).to("cpu")
                    # canvas=generate_img_canvas(model_output_detach,target_detach,args.image_size)
                    
                    save_image(visual_img_tensor, f"{experiment_dir}/sample-step={train_steps:07d}.png", nrow=visual_num, normalize=True, value_range=(-1, 1))
                    # canvas_path=os.path.join(experiment_dir,"canvas-"+f"step={train_steps:07d}.png")
                    # canvas.save(canvas_path)

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        epoch_loss=per_epoch_running_loss/len(loader)

        if epoch_loss<min_loss:
            min_loss=epoch_loss
            best_model_ckpt= {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }


    # 训练结束后保存ckpt
    try:
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/final.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            if best_model_ckpt:
                best_checkpoint_path = f"{checkpoint_dir}/best.pt"
                torch.save(best_model_ckpt, best_checkpoint_path)
                logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
        dist.barrier()
    except Exception as e:
        print("训练结束后保存ckpt 失败",e)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info(f"min_loss is {min_loss}")
    logger.info("Done!")

    log_path=f"{experiment_dir}/log.txt"
    visualize_log(log_path,experiment_dir)


    train_prefix="/home/tongji209/majiawei/Dit/dataset/train/source/eroded_"
    gt_prefix="/home/tongji209/majiawei/Dit/dataset/train/target/target_"

    sample_idx=[0]
    img_list=[]

    

    for i in sample_idx:
        yy=Image.open(train_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        img_list.append(yy)

    cond=torch.cat(tuple(img_list),0).to(device)


    sample_num=cond.shape[0]

    # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(sample_num, 4, latent_size, latent_size, device=device)
    z=vae.encode(cond).latent_dist.sample().mul_(0.18215)

    eval_model_kwargs = dict(y=cond)

    gt_list=[]
    for i in sample_idx:
        yy=Image.open(gt_prefix+f"{i}.jpg").convert("RGB")
        yy=transform(yy)
        yy=yy.unsqueeze(0)
        gt_list.append(yy)

    ggtt=torch.cat(tuple(gt_list),0).to(device)

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
        sample = vae.decode(sample / 0.18215).sample
        if sample_cnt% args.sample_visual_every == 0:
            visual_img_path=os.path.join(experiment_dir,"canvas-"+f"{sample_cnt:04d}.png")
            sample_compare=torch.cat((sample,ggtt),0).to(device)
            save_image(sample_compare, visual_img_path, nrow=4, normalize=True, value_range=(-1, 1))
        sample_loss=mean_flat((ggtt - sample) ** 2)
        with open(os.path.join(experiment_dir,"loss.txt"),"a") as f:
            f.write(f"step-{sample_cnt}-loss : ")
            for loss in sample_loss:
                f.write(f"{loss:.6f} " )
            f.write("\n")
        sample_cnt+=1

    img_name=f"generated.png"
    save_image(sample, os.path.join(experiment_dir,img_name), nrow=4, normalize=True, value_range=(-1, 1))
    save_gt_img(transform,experiment_dir,gt_prefix,sample_idx)
    save_train_img(transform,experiment_dir,train_prefix,sample_idx)
    

    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--gt-data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--visual-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None) #加载ckpt文件
    parser.add_argument("--learning-rate", type=float, default=1e-4) 
    parser.add_argument("--sample-visual-every", type=int, default=10)
    parser.add_argument("--predict-xstart", action='store_true', help="use predict xstart in diffusion training")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    args = parser.parse_args()
    main(args)
