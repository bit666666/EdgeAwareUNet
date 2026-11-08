import os
import time
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
import torch.nn.functional as F
from Net import *
from torch.optim.lr_scheduler import LambdaLR
from utils import *

def log_message(message, log_file="output.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message )  
h=256
w=512
lr=1e-4
batch_size=2
num_epochs = 350
max_patience = 25
r=4 
dataset='turbine_dataset_2'
if dataset=='turbine_dataset_2':
    num_classes = 2
elif dataset=='turbine_dataset_1':
    num_classes = 3
model_path='models_2'
output_path='output_2'
n=1
while n<=10:
    modelname=f"EdgeAwareUnet-{n}"
    model= EdgeAwareUnet(num_classes).to(device)
    dsloss=Dsloss(num_classes=num_classes,focal=True,loss_weight=[1.0,0.2,0.2],ouput_weight=[1.0,0.2,0.2,0.2,0.2]).to(device)
    celoss=nn.CrossEntropyLoss()
    file_out= f"{output_path}/{modelname}.txt"
    with open(file_out, "w") as f:
        f.write("")
    timestamp, python_seed, numpy_seed, torch_seed, cuda_seeds = get_and_save_random_seeds()
    with open(file_out, 'a', encoding="utf-8") as f:
        f.write(f"时间：{timestamp}, python种子：{python_seed}, numpy种子：{numpy_seed}, torch种子：{torch_seed}, cuda种子：{cuda_seeds}\n")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda epoch: (1 - epoch / num_epochs) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((h, w)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_label = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((h, w))
    ])
    transform_enhanced = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    train_dataset = WindTurbineDataset(root_dir=dataset,mode='train', transform_img=transform_img, transform_label=transform_label
                                  ,transform_enhanced=None)

    if len(train_dataset)%batch_size == 1:
        drop=True
    else:
        drop=False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=drop)
    test_dataset = WindTurbineDataset(root_dir=dataset,mode='test',transform_img=transform_img, transform_label=transform_label
                                  ,transform_enhanced=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,drop_last=False)


    train_num_samples = float(len(train_loader))
    val_num_samples = float(len(test_loader))
    patience_counter=0
    bestiou = 0.0
    for epoch in range(num_epochs):
        train_total_intersection = [0] * num_classes
        train_total_union = [0] * num_classes
        train_iou_list = [0.0] * num_classes
        train_num_list = [0] * num_classes
        val_total_intersection = [0] * num_classes
        val_total_union = [0] * num_classes
        val_iou_list = [0.0] * num_classes
        val_num_list = [0] * num_classes
        train_total_loss = 0.0
        val_total_loss = 0.0
        print('----------------------------------Epoch：{}/{}----------------------------------'.format(epoch + 1,
                                                                                                       num_epochs))
        time.sleep(0.1)
        model.train()
        for image, label_1D, label_3D in tqdm(train_loader):
            train_image = image.to(device)
            train_label_1D = label_1D.to(device)
            optimizer.zero_grad()
            output = model(train_image)
            
            train_dsloss = dsloss(output, train_label_1D)
            train_dsloss.backward()
            optimizer.step()
            train_total_loss += train_dsloss.item()
            
            output = F.softmax(output[0], dim=1)
            preds = torch.argmax(output, dim=1)

            intersection, union = calculate_classiou_batch(preds, train_label_1D, num_classes=num_classes)
            train_total_union = [a + b for a, b in zip(train_total_union, union)]
            train_total_intersection = [a + b for a, b in zip(train_total_intersection, intersection)]
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for image, label_1D, label_3D in tqdm(test_loader):
                val_image = image.to(device)
                val_label_1D = label_1D.to(device)
                output = model(val_image)
                val_dsloss = celoss(output, val_label_1D)
                val_total_loss += val_dsloss.item()
                output = F.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                intersection, union = calculate_classiou_batch(preds, val_label_1D, num_classes=num_classes)
                val_total_union = [a + b for a, b in zip(val_total_union, union)]
                val_total_intersection = [a + b for a, b in zip(val_total_intersection, intersection)]

            
            train_epoch_loss = train_total_loss / train_num_samples
            train_epoch_iou_list = [
                train_total_intersection[i] / train_total_union[i] if train_total_union[i] > 0 else 0.0 for i in
                range(num_classes)]
            train_epoch_iou_list = [round(x, r) for x in train_epoch_iou_list]

            val_epoch_loss = val_total_loss / val_num_samples
            val_epoch_iou_list = [val_total_intersection[i] / val_total_union[i] if val_total_union[i] > 0 else 0.0 for
                                  i in range(num_classes)]
            val_epoch_iou_list = [round(x, r) for x in val_epoch_iou_list]

        message1 = (
            f'Epoch：{epoch + 1}/{num_epochs}\nTrain---Loss: {train_epoch_loss:.4f},Class IoU: {train_epoch_iou_list}\n')
        log_message(message1, file_out)
        message2 = (
            f'Val---Loss: {val_epoch_loss:.4f},Class IoU: {val_epoch_iou_list},Avg: {np.round(np.mean(val_epoch_iou_list), r)}\n')
        log_message(message2, file_out)
        print(message1)
        print(message2)
        if np.mean(val_epoch_iou_list) > bestiou:
            patience_counter = 0
            torch.save(model, f'./{model_path}/{modelname}.pth')
            bestclassiou = val_epoch_iou_list
            bestiou = np.mean(bestclassiou)
        if np.mean(val_epoch_iou_list) < bestiou:
            patience_counter += 1
        if patience_counter >= max_patience:
            m = f'Epoch：{epoch + 1},Val---BestIoU: {bestiou:.4f},Class IoU: {bestclassiou},Avg: {np.round(np.mean(bestclassiou), r)},Early Stopping'
            log_message(m, file_out)
            break
    n = n + 1
    print(f'Val---BestIoU: {bestiou:.4f},Class IoU: {bestclassiou}')