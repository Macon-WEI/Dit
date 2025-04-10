import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torchvision.utils import save_image

# #################################################################################
# #                   Sine/Cosine Positional Embedding Functions                  #
# #################################################################################
# # https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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

class StrokeOrderEmbedder(nn.Module):
    def __init__(self,stroke_nums,hidden_size):
        super().__init__()
        self.stroke_nums=stroke_nums
        self.embedding_table=nn.Embedding(stroke_nums,hidden_size)

    def forward(self,stroke_orders):
        embeddings=self.embedding_table(stroke_orders)


        return embeddings

class UnicodeEmbedder(nn.Module):
    def __init__(self,unicode_nums,hidden_size):
        super().__init__()

        self.unicode_nums=unicode_nums
        self.embedding_table=nn.Embedding(unicode_nums,hidden_size)


    def forward(self,unicodes):

        embeddings=self.embedding_table(unicodes)

        return embeddings



class CondFusion(nn.Module):
    def __init__(self,img_size=256,img_channels=3,vae_size=32,vae_patch_size=2,depth=6,num_heads=8,mlp_ratio=3,hidden_size=512,unicode_nums=None,stroke_nums=None) -> None:
        super().__init__()


        num_patches=(vae_size//vae_patch_size)**2
        self.patch_size=img_size//(vae_size//vae_patch_size)

        # patchify
        self.patch_embed=PatchEmbed(img_size,self.patch_size,img_channels,hidden_size)

        # 位置编码
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,hidden_size),requires_grad=False)

        # cls token
        # 可能初始化的维度不是1，1，hidden——size
        self.cls_token=nn.Parameter(torch.randn(1,1,hidden_size))

        # Transformer Encoder
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_heads)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=depth)


        self.stroke_order_embedder=StrokeOrderEmbedder(stroke_nums,hidden_size)

        self.unicode_embedder=UnicodeEmbedder(unicode_nums,hidden_size)

        # self.stroke_layer_norm=nn.LayerNorm(hidden_size)
        # self.unicode_layer_norm=nn.LayerNorm(hidden_size)
        # self.fusion_layer_norm=nn.LayerNorm(hidden_size)


        self.uni_str_fusion=nn.Sequential(
            # nn.LayerNorm(hidden_size),      # 尝试消融？
            nn.Conv1d(hidden_size,hidden_size,kernel_size=2),
            nn.GELU(),
            )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5),cls_token=True,extra_tokens=1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)


        nn.init.normal_(self.stroke_order_embedder.embedding_table.weight, std=0.02)
        
        nn.init.normal_(self.unicode_embedder.embedding_table.weight, std=0.02)

        nn.init.kaiming_normal_(self.uni_str_fusion[0].weight, mode='fan_in', nonlinearity='linear')
        with torch.no_grad():
            self.uni_str_fusion[0].weight.data *= 0.8  # 经验性缩放

    def forward(self,x,unicode,stroke_order):
        batch_size=x.size(0)

        # patches=x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)

        # patches=patches.view(batch_size,self.)

        x=self.patch_embed(x)
        # print(self.cls_token.size())
        cls_tokens=self.cls_token.expand(batch_size,-1,-1)  # B,1,D
        embedding=torch.cat((cls_tokens,x),dim=1)   # B,N+1,D
        embedding=embedding+self.pos_embed  # B,N+1,D
        embedding=embedding.permute(1,0,2)

        output=self.encoder(embedding)


        # stroke_emb=self.stroke_order_embedder(stroke_order).unsqueeze(1)    # (N,1,D)
        stroke_emb=self.stroke_order_embedder(stroke_order)    # (N,1,D)


        # unicode_emb=self.unicode_embedder(unicode).unsqueeze(1)             # (N,1,D)

        # stroke_unicode_cat=torch.cat((unicode_emb,stroke_emb),dim=1)    # (N,2,D)

        # stroke_unicode_cat=stroke_unicode_cat.permute(0,2,1)    # (N,D,2)

        # stroke_unicode_fused=self.uni_str_fusion(stroke_unicode_cat)    # (N,D,1)

        # stroke_unicode_fused=stroke_unicode_fused.permute(0,2,1).squeeze(1)        # (N,D)


        cls_output=output[0]+stroke_emb

        return output[0],cls_output



# sample_idx=[19971,19972,19975,19976,19978]
# stroke_sample_idx=[2,1,3,3,2]

# img=torch.randint(0,256,(3,256,256),dtype=torch.float32)
# img=img.unsqueeze(0)
# print(img.size())
# img_encoder=ImageEncoder(
#     img_size=256,img_channels=3,vae_size=32,vae_patch_size=2,depth=6,head_nums=8,mlp_ratio=3,hidden_size=512
# )

# output=img_encoder(img)
# print(output.size())




import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL
from data.dataset import CharacterStrokePairDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

class PairedImageDataset(Dataset):
    def __init__(self,train_dir,gt_dir,transform=None) -> None:
        super().__init__()
        self.train_dir=train_dir
        self.gt_dir=gt_dir
        self.transform=transform

        self.train_images=sorted(os.listdir(train_dir))
        self.gt_images=sorted(os.listdir(gt_dir))

        assert len(self.train_images)==len(self.gt_images), "训练图像和gt图像数量不匹配"


    def __len__(self):
        return len(self.train_images)
    

    def __getitem__(self,idx):
        train_image_path=os.path.join(self.train_dir,self.train_images[idx])
        gt_image_path=os.path.join(self.gt_dir,self.gt_images[idx])

        train_image=Image.open(train_image_path)
        gt_image=Image.open(gt_image_path)

        transform_tmp=transforms.Compose([
        transforms.ToTensor(),
        ])

        print((transform_tmp(train_image)==-1).any())
        print(transform_tmp(train_image).size())

        train_image=Image.open(train_image_path).convert("RGB")
        gt_image=Image.open(gt_image_path).convert("RGB")

        
        print(transform_tmp(train_image).size())
        

        if self.transform:
            train_image=self.transform(train_image)
            gt_image=self.transform(gt_image)
        
        print("processed image",(train_image[2]==train_image[1]).all().item())

        return train_image,gt_image

def cosine_similarity_matrix(embeddings):
    # embeddings: [N, D]
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化
    sim_matrix = torch.mm(embeddings, embeddings.T)   # [N, N]
    return sim_matrix

def plot_heatmap(matrix, title="Similarity Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1)  # 余弦相似度范围 [0,1]
    plt.title(title)
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Index")
    plt.savefig("Cosine-Similarity-Matrix.png", dpi=300, bbox_inches='tight')
    plt.close()  # 保存后关闭图形，避免显示

device="cuda:0"

transform=transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

character_dir="/home/tongji209/majiawei/stroke_segmentation/train_debug/characters-1"
stroke_dir="/home/tongji209/majiawei/stroke_segmentation/train_debug/strokes-1"
csv_path="/home/tongji209/majiawei/stroke_segmentation/train_debug/stroke_data_1.csv"
dataset=CharacterStrokePairDataset(csv_path,
                 character_dir,
                 stroke_dir,
                 character_transform=transform,
                 stroke_transform=transform)


loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    # sampler=sampler,
    num_workers=1,
    collate_fn=lambda batch: {
        'character': torch.stack([x['character'] for x in batch]),
        'unicode': torch.stack([x['unicode'] for x in batch]),
        'stroke': torch.stack([x['stroke'] for x in batch]),
        'stroke_order': torch.tensor([x['stroke_order'] for x in batch]),
        'stroke_nums': torch.tensor([x['stroke_nums'] for x in batch]),
    },
    pin_memory=True,
    drop_last=True
)


sample_idx=[19971,19972,19975,19976,19978]
stroke_sample_idx=[2,1,3,3,2]

# img=torch.randint(0,256,(3,256,256),dtype=torch.float32)
# img=img.unsqueeze(0)
# print(img.size())
# img_encoder=ImageEncoder(
#     img_size=256,img_channels=3,vae_size=32,vae_patch_size=2,depth=6,head_nums=8,mlp_ratio=3,hidden_size=512
# )

vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-mse").to(device)

condfusion=CondFusion(img_size=256,img_channels=3,vae_size=32,vae_patch_size=2,depth=6,num_heads=8,mlp_ratio=3,hidden_size=512,stroke_nums=24,unicode_nums=50000).to(device)

# output=img_encoder(img)
# print(output.size())
hidden_size=512
stroke_order_embedder=StrokeOrderEmbedder(24,hidden_size).to(device)

unicode_embedder=UnicodeEmbedder(50000,hidden_size).to(device)

stroke_order_list=torch.tensor(range(24)).to(device)

stroke_emb=stroke_order_embedder(stroke_order_list)

sim_matrix = cosine_similarity_matrix(stroke_emb).detach().cpu().numpy()
plot_heatmap(sim_matrix, title="Cosine Similarity Matrix")
input()


for batch in loader:
    # logger.info(f"train_steps {train_steps}")
    characters=batch["character"].to(device)
    unicodes=batch["unicode"].to(device)
    strokes=batch["stroke"].to(device)
    stroke_orders=batch['stroke_order'].to(device)

    stroke_emb=stroke_order_embedder(stroke_orders)

    sim_matrix = cosine_similarity_matrix(stroke_emb).detach().cpu().numpy()
    plot_heatmap(sim_matrix, title="Cosine Similarity Matrix")
    input()

    # print(condfusion.encoder)

    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        strokes = vae.encode(strokes).latent_dist.sample().mul_(0.18215)

        characters_enc = vae.encode(characters).latent_dist.sample().mul_(0.18215)

    cls_tkn,cls_tkn_fused=condfusion(characters,unicodes,stroke_orders)

    print(cls_tkn.size(),cls_tkn_fused.size())


    A = cls_tkn
    B = cls_tkn_fused

    # 1. 归一化（按特征维度标准化，每个 tensor 内独立处理）
    def standardize(tensor):
        return (tensor - tensor.mean(axis=0)) / tensor.std(axis=0)

    A_norm = standardize(A).detach().cpu().numpy()
    B_norm = standardize(B).detach().cpu().numpy()

    # 2. 计算差异矩阵：直接差值
    diff = (A_norm - B_norm)

    # 3. 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(diff, cmap="coolwarm", center=0)
    plt.xlabel("Feature Dimension")
    plt.ylabel("Sample Index")
    plt.title("Heatmap of Difference (A_norm - B_norm)")
    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()  # 保存后关闭图形，避免显示

    # 4. 绘制散点图：例如展示第一个特征维度的对比
    plt.figure(figsize=(6, 6))
    plt.scatter(A_norm[:, 0], B_norm[:, 0], alpha=0.7)
    plt.plot([A_norm[:, 0].min(), A_norm[:, 0].max()],
            [A_norm[:, 0].min(), A_norm[:, 0].max()], 'r--')
    plt.xlabel("A_norm Feature 0")
    plt.ylabel("B_norm Feature 0")
    plt.title("Scatter Plot for Feature 0")
    plt.savefig("Scatter-Plot.png", dpi=300, bbox_inches='tight')
    plt.close()  # 保存后关闭图形，避免显示

    input()

    # # 5. PCA 降维，将差异矩阵降到二维进行可视化
    # pca = PCA(n_components=2)
    # diff_2d = pca.fit_transform(diff)
    # plt.figure(figsize=(8,6))
    # plt.scatter(diff_2d[:, 0], diff_2d[:, 1])
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("PCA of Difference Matrix")
    # plt.show()

    
    
    # save_image(train_batch, "./train.png", nrow=4, normalize=True, value_range=(-1, 1))
    # save_image(gt_batch, "./gt.png", nrow=4, normalize=True, value_range=(-1, 1))

    

    break  # 只查看一个批次的样本

# img=Image.open("/home/tongji209/majiawei/Dit/dataset/train/eroded/eroded_0.jpg")

# print(img.mode)
        
# z=torch.randn(2,4,32,32)
# print(z.shape)