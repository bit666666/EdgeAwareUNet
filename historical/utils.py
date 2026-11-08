from datetime import datetime
import random
from torch import nn
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

def compute_mean_std(dataloader):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_pixels = 0

    for images, _ ,_ in tqdm(dataloader):
        
        if images.dim() == 3:
            images = images.unsqueeze(1)

        images = images.float()  
        total_sum += images.sum(dim=[0, 2, 3])
        total_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += images.numel() / images.shape[1]  

    mean = total_sum / total_pixels
    std = torch.sqrt(total_squared_sum / total_pixels - mean ** 2)
    return mean, std

def calculate_classiou_batch(pred, label, num_classes=3):
    """Historical implementation retained for reproducibility."""
    total_intersection = [0] * num_classes
    total_union = [0] * num_classes
    p = pred.view(-1)
    r = label.view(-1)
    for sem_class in range(num_classes):  
        
        pred_inds = p == sem_class
        target_inds = r == sem_class
        if target_inds.long().sum().item() != 0:
            intersection = (pred_inds & target_inds).long().sum().item()
            pred_area = pred_inds.long().sum().item()
            target_area = target_inds.long().sum().item()
            union = pred_area + target_area - intersection
            total_intersection[sem_class] += intersection
            total_union[sem_class] += union
    return total_intersection,total_union

def get_and_save_random_seeds(seed=None):
    """Historical implementation retained for reproducibility."""
    if seed==None:
        python_seed = random.randint(1, 10000)
        numpy_seed = python_seed
        torch_seed = python_seed
        cuda_seeds = python_seed
    else:
        python_seed = seed
        numpy_seed = python_seed
        torch_seed = python_seed
        cuda_seeds = python_seed
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cuda_seeds)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return timestamp, python_seed, numpy_seed, torch_seed, cuda_seeds

def calculate_batch_tp_fp_fn(pred, target, num_classes=3):
    """Historical implementation retained for reproducibility."""
    
    assert pred.shape == target.shape, "pred和target形状必须一致"
    assert pred.dtype in (torch.long, torch.int64), "pred必须是整数类型"
    assert target.dtype in (torch.long, torch.int64), "target必须是整数类型"

    
    if num_classes is None:
        num_classes = torch.max(target).item() + 1  

    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    
    tp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fn = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)

    
    for cls in range(num_classes):
        
        tp[cls] = ((pred_flat == cls) & (target_flat == cls)).sum().float()
        
        fp[cls] = ((pred_flat == cls) & (target_flat != cls)).sum().float()
        
        fn[cls] = ((pred_flat != cls) & (target_flat == cls)).sum().float()

    return tp, fp, fn
class FocalLoss(nn.Module):
    """Historical implementation retained for reproducibility."""
    def __init__(self, num_classes=5, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  

        
        return torch.mean(focal_loss)
class DsCEloss(nn.Module):
    """Historical implementation retained for reproducibility."""

    def __init__(self, num_classes=5,focal=False,ouput_weight=[1.0,0.2,0.2,0.2,0.2]):
        super(DsCEloss, self).__init__()
        self.num_classes = num_classes
        self.weight = ouput_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal = focal
        self.focal_loss = FocalLoss(num_classes=num_classes)
    def forward(self, outputs, targets):
        """Historical implementation retained for reproducibility."""
        ce_total = 0.0
        focal_total = 0.0
        for i in range(len(outputs)):
            ce_total += self.weight[i] * self.ce_loss(outputs[i], targets)
            focal_total += self.weight[i] * self.focal_loss(outputs[i], targets)
        if self.focal:
            loss_total = ce_total + 0.4*focal_total
        else:
            loss_total = ce_total
        return loss_total
class JaccardLoss(nn.Module):
    def __init__(self, num_classes, weight=None, smooth=1e-5, reduction='mean'):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction
        self.weight = weight if weight is not None else [1.0] * num_classes

    def forward(self, inputs, targets):
        if not isinstance(inputs, list):
            inputs = [inputs]
        total_loss = 0.0
        for input in inputs:
            
            input_probs = F.softmax(input, dim=1)
            input_labels = torch.argmax(input_probs, dim=1)  # [B, H, W]
            target_labels = targets

            
            layer_loss = 0.0
            for c in range(self.num_classes):
                input_c = (input_labels == c).float()
                label_c = (target_labels == c).float()
                intersection = torch.sum(input_c * label_c)
                union = torch.sum(input_c) + torch.sum(label_c) - intersection
                jaccard = (intersection + self.smooth) / (union + self.smooth)
                layer_loss += self.weight[c] * (1.0 - jaccard)
            total_loss += layer_loss

        return total_loss / (len(inputs) * self.num_classes)
class DiceLoss(nn.Module):
    """Historical implementation retained for reproducibility."""

    def __init__(self, num_classes=5, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """Historical implementation retained for reproducibility."""
        if not isinstance(inputs, list):
            inputs = [inputs]

        total_loss = 0.0
        for input in inputs:
            one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            input_probs = F.softmax(input, dim=1)

            intersection = torch.sum(input_probs * one_hot_targets, dim=(0, 2, 3))
            union = torch.sum(input_probs + one_hot_targets, dim=(0, 2, 3))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_loss += 1 - torch.mean(dice)

        return total_loss / len(inputs)
class Dsloss(nn.Module):
    """Historical implementation retained for reproducibility."""
    def __init__(self, num_classes=3,focal=False,loss_weight=[1.0,0.2,0.2],ouput_weight=[1.0,0.2,0.2,0.2,0.2]):
        super(Dsloss, self).__init__()
        self.num_classes = num_classes
        self.ouput_weight = ouput_weight
        self.loss_weight = loss_weight
        self.focal = focal

        self.ce_loss = nn.CrossEntropyLoss()
        self.jaccard_loss=JaccardLoss(num_classes=num_classes, smooth=1e-5)
        self.dice_loss=DiceLoss(num_classes=num_classes, smooth=1e-5)
        self.focal_loss=FocalLoss(num_classes=num_classes)

    def forward(self, outputs, targets):
        """Historical implementation retained for reproducibility."""
        if not isinstance(outputs, list):
            outputs = [outputs]
        ce_total = 0.0
        jaccard_total = 0.0
        dice_total = 0.0
        focal_total = 0.0
        for i in range(len(outputs)):
            focal_total += self.ouput_weight[i] * self.focal_loss(outputs[i], targets)

            ce_total += self.ouput_weight[i] * self.ce_loss(outputs[i], targets)
            jaccard_total += self.ouput_weight[i] * self.jaccard_loss(outputs[i], targets)
            dice_total += self.ouput_weight[i] * self.dice_loss(outputs[i], targets)
        if self.focal:
            loss_total = self.loss_weight[0]*ce_total + self.loss_weight[1]*jaccard_total + self.loss_weight[2]*dice_total + 0.4*focal_total
        else:
            loss_total = self.loss_weight[0]*ce_total + self.loss_weight[1]*jaccard_total + self.loss_weight[2]*dice_total
        return loss_total
class WindTurbineDataset(Dataset):
    def __init__(self, root_dir="turbine_dataset_1", mode='train', transform_img=None, transform_label=None,
                 transform_enhanced=None):
        self.root_dir =f'./{root_dir}'
        self.mode = mode
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.transform_enhanced = transform_enhanced
        self.image_dir = os.path.join(root_dir, mode, 'img')
        self.label_1D_dir = os.path.join(root_dir, mode, 'label_1D')
        self.label_3D_dir = os.path.join(root_dir, mode, 'label_3D')

        self.image_basenames = []
        valid_extensions = {'.jpg', '.jpeg', '.png','.JPG'}  
        for f in sorted(os.listdir(self.image_dir)):
            filename, ext = os.path.splitext(f)
            if ext.lower() in valid_extensions:  
                self.image_basenames.append(filename)

        
        self.check_data_integrity()

    def __len__(self):
        return len(self.image_basenames)

    def __getitem__(self, idx):
        basename = self.image_basenames[idx]
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png','.JPG']:
            path = os.path.join(self.image_dir, f"{basename}{ext}")
            if os.path.exists(path):
                image_path = path
                break
        if image_path is None:
            raise FileNotFoundError(f"找不到图像文件: {basename} (可能的扩展名: .jpg, .jpeg, .png, .JPG)")

        label_1D_path = os.path.join(self.label_1D_dir, f"{basename}.png")
        label_3D_path = os.path.join(self.label_3D_dir, f"{basename}.png")

        
        image = Image.open(image_path).convert('RGB')
        label_1D = Image.open(label_1D_path)
        label_3D = Image.open(label_3D_path).convert('RGB')

        
        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_label:
            label_1D = self.transform_label(label_1D)
            label_1D = label_1D.squeeze(0)
            label_3D = self.transform_label(label_3D)

        if self.mode == 'train' and self.transform_enhanced:
            if label_1D.dim() == 2:
                label_1D = label_1D.unsqueeze(0)
            stacked = torch.cat([image, label_1D, label_3D], dim=0)
            stacked = self.transform_enhanced(stacked)
            image = stacked[:3, :, :]
            label_1D = stacked[3, :, :]
            label_1D = label_1D.squeeze(0).long()
            label_3D = stacked[4:, :, :]
        return image, label_1D.long(), label_3D

    def check_data_integrity(self):
        """Historical implementation retained for reproducibility."""
        missing_images = []
        missing_label_1D = []
        missing_label_3D = []

        for basename in self.image_basenames:
            
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                path = os.path.join(self.image_dir, f"{basename}{ext}")
                if os.path.exists(path):
                    image_path = path

                    label_1d_path = os.path.join(self.label_1D_dir, f"{basename}.png")
                    if not os.path.exists(label_1d_path):
                        missing_label_1D.append(basename)

                    label_3d_path = os.path.join(self.label_3D_dir, f"{basename}.png")
                    if not os.path.exists(label_3d_path):
                        missing_label_3D.append(basename)
                    break
            if image_path is None:
                raise FileNotFoundError(f"找不到图像文件: {basename} (可能的扩展名: .jpg, .jpeg, .png)")

        if missing_images:
            print(f"警告: 找不到以下图像文件: {missing_images}")
        if missing_label_1D:
            print(f"警告: 找不到以下label_1D文件: {missing_label_1D}")
        if missing_label_3D:
            print(f"警告: 找不到以下label_3D文件: {missing_label_3D}")
if __name__ == "__main__":
    # transform_img = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 512),antialias=False),
    #     # transforms.Normalize(mean=[0.5139, 0.5311, 0.4513], std=[0.1839, 0.1465, 0.1418]),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # transform_label = transforms.Compose([
    #     transforms.PILToTensor(),
    #     transforms.Resize((256, 512),antialias=False)
    # ])
    # transform_enhanced= transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    # ])
    # dataset1=WindTurbineDataset(mode='train',transform_img=transform_img,transform_label=transform_label
    #                             ,transform_enhanced=None)
    # train_loader = DataLoader(dataset1, batch_size=8, shuffle=True, drop_last=False)
    # mean,std = compute_mean_std(train_loader)
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")
    for i in range(0,2):
        print(i)