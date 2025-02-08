import os
import argparse
from tqdm import tqdm
import torch
from datetime import datetime
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from dataloader import ConvNeXTDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier train parameters")
    parser.add_argument("--vector-path", type=str, required=True, help="pixel latent vector save path")
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--categories", type=int, required=True, help="number of categories")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--evaluation", action="store_true")
    return parser.parse_args()

def merge_excel_files(excel_folder_path, sample_ratio=0.1):
    # 获取文件夹中所有 Excel 文件
    excel_files = [f for f in os.listdir(excel_folder_path) if f.endswith(('.csv'))]
    
    # 用于存储所有数据的列表
    all_data = []

    for file in excel_files:
        file_path = os.path.join(excel_folder_path, file)
        # 读取 Excel 文件，没有表头
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    # 合并所有数据
    merged_data = pd.concat(all_data, ignore_index=True)
    
    # 从合并的数据中随机取 10%
    sampled_data = merged_data.sample(frac=sample_ratio, random_state=42)
    
    return sampled_data

def load_data(excel_folder, num_inference_steps):
    print("loading data...")
    merged_data = merge_excel_files(excel_folder)
    X = merged_data.iloc[:, 1:num_inference_steps+1].values.astype(np.float32)
    y = merged_data.iloc[:, num_inference_steps+1].values.astype(np.int64)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    conv_train_dataset = ConvNeXTDataset(X_train, y_train)
    conv_val_dataset = ConvNeXTDataset(X_val, y_val)
    conv_test_dataset = ConvNeXTDataset(X_test, y_test)

    conv_train_loader = DataLoader(conv_train_dataset, batch_size=256, shuffle=True)
    conv_val_loader = DataLoader(conv_val_dataset, batch_size=256, shuffle=False)
    conv_test_loader = DataLoader(conv_test_dataset, batch_size=256, shuffle=False)
    return conv_train_loader, conv_val_loader, conv_test_loader

def define_model(categories, num_inference_steps):
    # 加载 ConvNeXt-Tiny
    model = create_model('convnext_tiny', pretrained=False, num_classes=categories)
    # 修改输入通道
    model.stem[0] = nn.Conv2d(num_inference_steps, 96, kernel_size=4, stride=4, bias=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    # 设置csv记录
    current_date = datetime.now().strftime("%y_%m_%d")
    csv_file = f'train_save/loss/{current_date}/convnext_loss.csv'
    header = ['epoch', 'loss']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f'train_save/loss/{current_date}', exist_ok=True)
        
    # 写入 CSV 文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入列名
        writer.writerow(header)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), 'train_save/convnext_tiny/convnext_tiny.pth')
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, running_loss/len(train_loader)])

def eval_model(model, test_loader):
    all_labels = []
    all_preds = []
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return all_preds, all_labels

if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    conv_train_loader, conv_val_loader, conv_test_loader = load_data(args.vector_path, args.num_inference_steps)
    model, criterion, optimizer = define_model(args.categories, args.num_inference_steps)
    train_model(model=model, train_loader=conv_train_loader, val_loader=conv_val_loader, criterion=criterion, optimizer=optimizer, epochs=args.epochs)
    if args.evaluation == True:
        eval_model(model=model, test_loader=conv_test_loader)