import os

from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn.functional as F
from utils import *
h=256
w=512
n=1
while n<=1:
    name=f'DeepLabV3Plus-{n}'
    path='models_1'
    dataset='turbine_dataset_1'
    num_classes=3
    model=torch.load(f'./{path}/{name}.pth')
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((h, w)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_label = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((h, w))
    ])

    test_dataset = WindTurbineDataset(root_dir=dataset,mode='test',transform_img=transform_img,transform_label=transform_label)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,drop_last=False)
    val_num_samples=len(test_loader)
    
    device = 'cuda'
    
    color_map = {
        0: [0, 0, 0],  
        1: [125, 125,125],
        2: [254, 254,254],
    }
    iou_list = [0.0,0.0,0.0]
    num_list=[0,0,0]
    val_total_iou=0.0
    val_iou_list=[0.0,0.0,0.0]
    total_tp = None
    total_fp = None
    total_fn = None
    val_total_intersection = [0] * num_classes
    val_total_union = [0] * num_classes
    with torch.no_grad():
        for i, (images,label,Label) in enumerate(test_loader):
            images = images.to(device)
            label = label.to(device)
            outputs = model(images)
            if type(outputs)==list:
                outputs=outputs[0]
            outputs=F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            intersection, union = calculate_classiou_batch(preds, label, num_classes=num_classes)
            batch_tp, batch_fp, batch_fn = calculate_batch_tp_fp_fn(preds, label, num_classes=num_classes)

            if total_tp is None:
                total_tp = batch_tp
                total_fp = batch_fp
                total_fn = batch_fn
            else:
                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += batch_fn
            val_total_union = [a + b for a, b in zip(val_total_union, union)]
            val_total_intersection = [a + b for a, b in zip(val_total_intersection, intersection)]
            val_epoch_iou_list = [val_total_intersection[i] / val_total_union[i] if val_total_union[i] > 0 else 0.0 for
                                  i in range(num_classes)]
            val_epoch_iou_list = [round(x, 4) for x in val_epoch_iou_list]
            preds = preds.cpu()
            preds = preds.numpy()
            pre_image = preds[0]


            color_image = np.zeros((pre_image.shape[0], pre_image.shape[1], 3), dtype=np.uint8) 
            for label_idx in range(num_classes):  
                color_image[pre_image == label_idx] = color_map[label_idx]  

            label_image = Label[0]  
            label_image = np.transpose(label_image, (1, 2, 0))
            # plt.subplot(1, 2, 1)
            # plt.imshow(label_image)
            # plt.subplot(1, 2, 2)
            plt.imshow(color_image)
            path=f'./pic/dataset_1_{name[:-2]}'
            if not os.path.exists(path):
                
                os.makedirs(path, exist_ok=True)  
            save_path4 = path+f'/sample_{i}.png'
            plt.axis('off')
            plt.savefig(save_path4,dpi=300, bbox_inches='tight')
            plt.close()
        eps = 1e-6
        precision = (total_tp + eps) / (total_tp + total_fp + eps)
        recall = (total_tp + eps) / (total_tp + total_fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()
        
        
        

        message2 = (
            f'Class IoU: {val_epoch_iou_list},Avg: {np.round(np.mean(val_epoch_iou_list), 4)}')
        print(message2)
        n+=1

