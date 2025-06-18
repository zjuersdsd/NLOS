from torch.utils.data import DataLoader, random_split
from utils.dataloder import get_dataloader, AudioDataset
from model.ResNet import ResNet18_2D, ResNet34_2D
import random
import numpy as np
import torch
import os
import argparse

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_data(scene_path, test_size=0.25, batch_size=32, num_channels=2):
    # Load the dataset for the single scene (not the DataLoader)
    dataset = AudioDataset(scene_path,num_channels=num_channels)

    # Calculate the number of test samples (25%)
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len

    # Split into training and test datasets
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # Return the DataLoader for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(scene_path="../data/nlos_data_segmented_two_channel/",
          scene_name="scene1",
          num_epochs=10,
          batch_size=32,
          num_channels=2,
          order_is_trainable=None,
          order=None,
          transform=None,
          backbone="resnet18",
          model_save_path="../checkpoints/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transform == 'frft':
        if order_is_trainable == True:
            model_name = "model_" + backbone + "_train_on_" + scene_name + "_transform_" \
                         + transform + '_order_is_trainable_' + str(order_is_trainable)
        else:
            model_name = "model_" + backbone + "_train_on_" + scene_name + "_transform_" \
                         + transform + '_order_is_trainable_' + str(order_is_trainable) + '_order_' + str(order)
    else:
        model_name = "model_" + backbone + "_train_on_" + scene_name + "_transform_" + transform
    model_save_name = os.path.join(model_save_path, model_name)

    if backbone == "resnet18":
        model = ResNet18_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order
                            ).to(device)
    elif backbone == "resnet34":
        model = ResNet34_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order
                            ).to(device)

    scene_load_path = os.path.join(scene_path, scene_name)
    train_loader, test_loader = split_data(scene_load_path, batch_size=batch_size, num_channels=num_channels)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for waveform, labels in train_loader:
            waveform, labels = waveform.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveform)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            model_best = model
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_name)
            print(f"New best model saved with loss {best_loss:.4f}")

    # After training, we test the model
    model_best.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for waveform, labels in test_loader:
            waveform, labels = waveform.to(device), labels.to(device)
            outputs = model_best(waveform)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and Test identical-scene')
    parser.add_argument('--data_path', default='./data/nlos_data_segmented_two_channel/', help='DATA_ROOT')
    parser.add_argument('--scene_name', default='scene1_corridor')
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--order_is_trainable', action='store_true')
    parser.add_argument('--order', default=0.5)
    parser.add_argument('--feature_extraction_layer', default='frft')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--model_save_path', default='./checkpoints_identical/', help='Checkpoints_ROOT')
    # args parse
    args = parser.parse_args()
    data_path, scene_name, num_epochs, batch_size, num_channels = args.data_path, args.scene_name, \
        args.num_epochs, args.batch_size, args.num_channels
    order_is_trainable, order, feature_extraction_layer, backbone, model_save_path = \
        args.order_is_trainable, args.order, args.feature_extraction_layer, args.backbone, args.model_save_path

    main(scene_path=data_path,
          scene_name=scene_name,
          num_epochs=num_epochs,
          batch_size=batch_size,
          num_channels=num_channels,
          order_is_trainable=order_is_trainable,
          order=order,
          transform=feature_extraction_layer,
          backbone=backbone,
          model_save_path=model_save_path)
