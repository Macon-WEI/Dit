import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image
from glob import glob

sys.path.append(os.path.abspath("./train_options"))
from train_options.train_inpaint import center_crop_arr

# 定义图像转换
# transform = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

# 加载所有真实图像
real_image_paths = []
target_root_dir = '/home/tongji209/majiawei/Dit/dataset/test/task1/target'  # 真实图像的根目录

for subdir, _, files in os.walk(target_root_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            real_image_paths.append(os.path.join(subdir, file))


sample_image_paths = []
sample_root_dir = '/home/tongji209/majiawei/Dit/dataset/real-sample/sample-000'  # 真实图像的根目录


for subdir, _, files in os.walk(target_root_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            sample_image_paths.append(os.path.join(subdir, file))


# 创建Dataset和DataLoader
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 这里的0只是占位符，没有实际意义

real_dataset = CustomImageDataset(real_image_paths, transform=transform)




# real_dataset = ImageFolder(root=target_root_dir, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)

# 加载所有生成图像
# fake_dataset = ImageFolder(root=sample_root_dir, transform=transform)
fake_dataset = CustomImageDataset(sample_image_paths, transform=transform)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# 函数来提取特征
def get_features(loader, model, device):
    features = []
    for batch in loader:
        images, _ = batch
        images = images.to(device)
        with torch.no_grad():
            pred = model(images)
        features.append(pred.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

# 计算FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def save_result(target_image_paths,sample_image_paths,fid_result_dir,fid_result):
    fid_index = len(glob(f"{fid_result_dir}/*"))
    fid_dir = f"{fid_result_dir}/{fid_index:03d}"
    if not os.path.exists(fid_dir):
        os.makedirs(fid_dir,exist_ok=True)

    with open(os.path.join(fid_dir,"fid_result.txt"),"w") as f:
        f.write(f'target_image_path: {target_image_paths}\n')
        f.write(f'sample_image_path: {sample_image_paths}\n')
        f.write(f'Overall FID: {fid_result}')


# 下载并加载InceptionV3模型
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # 使用InceptionV3的特征提取部分
inception_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception_model.to(device)

# 提取真实和生成图像的特征
real_features = get_features(real_loader, inception_model, device)
fake_features = get_features(fake_loader, inception_model, device)

# 计算均值和协方差
mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)
mu_fake = np.mean(fake_features, axis=0)
sigma_fake = np.cov(fake_features, rowvar=False)

# 计算FID
fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

fid_result_dir="./FID"

if not os.path.exists(fid_result_dir):
    os.makedirs(fid_result_dir,exist_ok=True)



print('Overall FID:', fid)
save_result(target_root_dir,sample_root_dir,fid_result_dir,fid)