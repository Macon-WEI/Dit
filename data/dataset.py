import os
from torch.utils.data import Dataset
from PIL import Image
# import torchvision.transforms as transforms

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

        train_image=Image.open(train_image_path).convert("RGB")
        gt_image=Image.open(gt_image_path).convert("RGB")

        if self.transform:
            train_image=self.transform(train_image)
            gt_image=self.transform(gt_image)

        return train_image,gt_image

