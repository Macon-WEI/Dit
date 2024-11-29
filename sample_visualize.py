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
from models import DiT_models
import argparse
import os


def generate_img_canvas(model_out_tens,img_size):
    import torch
    import numpy as np
    from PIL import Image

    # 定义一个函数将张量转换为图片
    def tensor_to_image(tensor):
        # 将张量从 [3, 256, 256] 转换为 [256, 256, 3]
        image = tensor.permute(1, 2, 0).cpu().numpy()
        # 确保图片的值在 [0, 1] 范围内
        image = np.clip(image, 0, 1)
        # 转换为 [0, 255] 范围内的 uint8 类型
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    # 将张量转换为图片
    # image1 = tensor_to_image(ten)
    # image2 = tensor_to_image(tensor2)

    img_nums_max = 8
    idx_max=min(model_out_tens.shape[0],img_nums_max)



    spacing=20

    # 创建一个新的大画布来绘制两个图像
    canvas_width = img_size
    canvas_height = idx_max*img_size+spacing*(idx_max-1)
    canvas = Image.new('RGB', (canvas_width, canvas_height))
    for idx in range(idx_max):
        # print(model_out_tens[idx].shape)
        image1=tensor_to_image(model_out_tens[idx])
        # image2=tensor_to_image(target_tens[idx])
        # print("image1.shape",image1.size)
        # print("image2.shape",image2.size)

        canvas.paste(image1,(0,idx*(img_size+spacing)))
        # canvas.paste(image2,(img_size+spacing,idx*(img_size+spacing)))


    # 将两个图像粘贴到画布上
    # canvas.paste(image1, (0, 0))
    # canvas.paste(image2, (256+spacing, 0))

    # 保存画布
    # canvas.save('combined_image.png')
    return canvas


def main(args):
    print(args.savedir)
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    #print(ckpt_path)
    state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    model.load_state_dict(state_dict,False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"/public/home/acr0vd9ik6/project/DiT/fast-DiT/sd-vae-ft-mse").to(device)

    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = [1]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    cnt=0
    for samples in diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    ):
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # with torch.no_grad():
        #     samples = vae.decode(samples / 0.18215).sample
        #     canvas=generate_img_canvas(samples,args.image_size)
        #     canvas_path=os.path.join("/public/home/acr0vd9ik6/project/DiT/fast-DiT/sample_result/sample-visualize","canvas-"+f"{cnt:04d}.png")
        #     canvas.save(canvas_path)
        samples = vae.decode(samples / 0.18215).sample
        canvas_path=os.path.join("/public/home/acr0vd9ik6/project/DiT/fast-DiT/sample_result/sample-visualize","canvas-"+f"{cnt:04d}.png")
        save_image(samples, canvas_path, nrow=4, normalize=True, value_range=(-1, 1))
        cnt+=1
        


    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # # Save and display images:
    # #save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    # save_image(samples, args.savedir, nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--savedir", type=str, default="sample.png")
    args = parser.parse_args()
    main(args)
