import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloder import get_dataloader, AudioDataset
import argparse
from model.ResNet import ResNet18_2D, ResNet34_2D
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os
import time  # 导入time模块

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_accuracy_curve(acc_list, save_path=None):
    """
    绘制测试集准确率随 epoch 变化的曲线（支持矢量图格式）
    :param acc_list: 每个 epoch 的准确率列表（单位：百分数）
    :param save_path: 若指定路径，则保存图片（推荐.svg/.pdf）；否则直接展示
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(acc_list) + 1), acc_list, marker='o', color='g')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Accuracy curve saved to {save_path}")
    else:
        plt.show()

def plot_loss_curve(loss_list, save_path=None):
    """
    绘制训练过程中 loss 随 epoch 变化的曲线（支持矢量图格式）
    :param loss_list: 每个 epoch 的平均 loss（列表）
    :param save_path: 若指定路径，则保存图片（推荐.svg/.pdf）；否则直接展示
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='svg')  # 或 format='pdf'
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()

def evaluate_accuracy(model, dataloader, device):
    """
    计算模型在测试集上的准确率
    :param model: 已训练的模型
    :param dataloader: 测试集 DataLoader
    :param device: CPU 或 GPU
    :return: 准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveform, labels in dataloader:
            waveform, labels = waveform.to(device), labels.to(device)
            outputs = model(waveform)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total


def train(scene_path="../data/nlos_data_segmented_two_channel/",
          scene_name="scene1",
          num_epochs=10,
          batch_size=32,
          num_channels=2,
          order_is_trainable=None,
          order=None,
          transform=None,
          backbone="resnet18",
          model_save_path="../checkpoints_cross/",
          only_use_real=None,
          ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_str = "_".join(scene_name)  # 将多个场景合并为字符串
    if transform == 'frft':
        if order_is_trainable:
            model_name = "model_" + backbone + "_train_on_" + scene_str+ "_transform_" \
                        + transform + '_order_is_trainable_' + str(order_is_trainable) + '_only_use_real_' +str(only_use_real)
        else:
            model_name = "model_" + backbone + "_train_on_" + scene_str + "_transform_" \
                         + transform + '_order_is_trainable_' + str(order_is_trainable) + '_order_' + str(order)
    else:
        model_name = "model_" + backbone + "_train_on_" + scene_str + "_transform_" + transform
    model_save_name = os.path.join(model_save_path, model_name)
    print("model save name :", model_save_name)
    if backbone== "resnet18":
        model = ResNet18_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order,
                            only_use_real=only_use_real,
                            ).to(device)
    elif backbone == "resnet34":
        model = ResNet34_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order,
                            only_use_real=only_use_real,
                            ).to(device)

    scene_load_path1 = os.path.join(scene_path, scene_name[0])
    dataset1 = AudioDataset(scene_load_path1, num_channels=num_channels, normalize=True)
    scene_load_path2 = os.path.join(scene_path, scene_name[1])
    dataset2 = AudioDataset(scene_load_path2, num_channels=num_channels, normalize=True)
    dataset = ConcatDataset([dataset1, dataset2])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')  # Track the best loss

    model.train()
    total_time = 0  # 记录所有 epoch 总时间
    loss_list = []  # 用于存储每个 epoch 的平均 loss
    acc_list = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 记录每个 epoch 的开始时间

        total_loss = 0.0
        for waveform, labels in train_loader:
            waveform, labels = waveform.to(device), labels.to(device)  # Move to GPU

            optimizer.zero_grad()
            outputs = model(waveform)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_end_time = time.time()  # 记录每个 epoch 的结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算每个 epoch 的时间
        total_time += epoch_duration

        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f} seconds')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_name)
            print(f"New best model saved with loss {best_loss:.4f}")

            # 加载测试集
    plot_loss_curve(loss_list, save_path="./images/"+model_name + "_loss_curve.svg")
    avg_epoch_time = total_time / num_epochs  # 计算平均每个 epoch 的时间
    print(f'Average time per epoch: {avg_epoch_time:.2f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cross-scene')
    parser.add_argument('--data_path', default='./data/nlos_data_segmented_two_channel/', help='DATA_ROOT')
    parser.add_argument('--train_scene_name', default=['scene2_hall', 'scene3_811'],nargs='+')
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--order_is_trainable', action='store_true')
    parser.add_argument('--only_use_real', action='store_true')
    parser.add_argument('--order', default=0.5, type=float)
    parser.add_argument('--feature_extraction_layer', default='frft')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--model_save_path', default='./checkpoints_revise/', help='Checkpoints_ROOT')

    # args parse
    args = parser.parse_args()
    data_path, train_scene_name, num_epochs, batch_size, num_channels, only_use_real = args.data_path, args.train_scene_name, \
        args.num_epochs, args.batch_size, args.num_channels, args.only_use_real
    order_is_trainable, order, feature_extraction_layer, backbone, model_save_path = \
        args.order_is_trainable, args.order, args.feature_extraction_layer, args.backbone, args.model_save_path

    train(scene_path=data_path,
          scene_name=train_scene_name,
          num_epochs=num_epochs,
          batch_size=batch_size,
          num_channels=num_channels,
          order_is_trainable=order_is_trainable,
          order=order,
          transform=feature_extraction_layer,
          backbone=backbone,
          model_save_path=model_save_path,
          only_use_real=only_use_real,
          )
