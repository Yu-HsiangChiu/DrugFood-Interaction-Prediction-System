import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import os
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot') 

DATA_DIR = 'dataset_augmented' 
NUM_EPOCHS = 20        
BATCH_SIZE = 16        
LEARNING_RATE = 0.001  
K_FOLDS = 5           

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")


base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=base_transforms)
class_names = full_dataset.classes
print(f"偵測到 {len(class_names)} 個類別: {class_names}")
print(f"總影像數量: {len(full_dataset)} 張")


def get_fresh_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 
    return model.to(device)


skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
targets = full_dataset.targets # 取得所有圖片的標籤用於分層

fold_results = {}
global_best_acc = 0.0
global_best_model_wts = None

print("\n開始 5-Fold 交叉驗證訓練")
print('=' * 50)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n進入 Fold {fold + 1}/{K_FOLDS}")
    print('-' * 30)
    
    
    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_sub), 'val': len(val_sub)}
    
    
    model = get_fresh_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    
    best_fold_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f'Fold {fold+1} - Epoch {epoch+1}/{NUM_EPOCHS}', end='\r')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_fold_acc:
                best_fold_acc = epoch_acc
                
                if epoch_acc > global_best_acc:
                    global_best_acc = epoch_acc
                    global_best_model_wts = copy.deepcopy(model.state_dict())
                    
    print(f'\n🏆 Fold {fold + 1} 完成 | 最佳驗證準確率: {best_fold_acc:.4f}')
    fold_results[f'Fold_{fold+1}'] = {'acc': best_fold_acc.item(), 'history': history}


print("\n" + "=" * 50)
print("5-Fold 交叉驗證最終報告")
print("=" * 50)
avg_acc = np.mean([res['acc'] for res in fold_results.values()])
for fold_name, res in fold_results.items():
    print(f"{fold_name} 準確率: {res['acc']:.4f}")
print(f"\n5-Fold 平均準確率: {avg_acc:.4f}")
print(f"最高準確率 (儲存為模型): {global_best_acc:.4f}")

torch.save(global_best_model_wts, 'best_resnet50_food_kfold.pth')
print("最佳模型已儲存為 resnet50_food_kfold.pth")


print("\n正在繪製平均學習曲線")
plt.figure(figsize=(12, 5))

avg_train_acc = np.mean([res['history']['train_acc'] for res in fold_results.values()], axis=0)
avg_val_acc = np.mean([res['history']['val_acc'] for res in fold_results.values()], axis=0)
avg_train_loss = np.mean([res['history']['train_loss'] for res in fold_results.values()], axis=0)
avg_val_loss = np.mean([res['history']['val_loss'] for res in fold_results.values()], axis=0)

plt.subplot(1, 2, 1)
plt.plot(avg_train_acc, label='Average Train Acc')
plt.plot(avg_val_acc, label='Average Val Acc', linestyle='--')
plt.title('5-Fold Average Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_train_loss, label='Average Train Loss')
plt.plot(avg_val_loss, label='Average Val Loss', linestyle='--')
plt.title('5-Fold Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('kfold_training_result.png', dpi=300)
plt.show()
print("平均訓練曲線已存為 kfold_training_result.png")