import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载所有真实图像
real_image_paths = []
real_root_dir = 'path_to_real_images'  # 真实图像的根目录

for subdir, _, files in os.walk(real_root_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            real_image_paths.append(os.path.join(subdir, file))

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
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)

# 加载所有生成图像
fake_dataset = ImageFolder(root='path_to_fake_images', transform=transform)
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
print('Overall FID:', fid)
