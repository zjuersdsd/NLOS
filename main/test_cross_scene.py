import torch
from utils.dataloder import get_dataloader
from model.ResNet import ResNet18_2D, ResNet34_2D
import os
import argparse

def test(test_scene_path="../data/nlos_data_segmented_two_channel/",
         test_scene_names=None,
         train_scene_name="scene1",
         batch_size =32,
         num_channels=2,
         order_is_trainable = True,
         order = 0.25,
         transform=None,
         backbone="resnet18",
         model_load_path="../checkpoints_cross/",
         only_use_real=None
         ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(order_is_trainable)
    print(f'Trained on {train_scene_name}')
    scene_str = "_".join(train_scene_name)  # 将多个场景合并为字
    if transform == 'frft':
        if order_is_trainable:
            model_name = "model_" + backbone + "_train_on_" + scene_str + "_transform_" \
                         + transform + '_order_is_trainable_' + str(order_is_trainable)+'_only_use_real_' +str(only_use_real)
        else:
            model_name = "model_" + backbone + "_train_on_" + scene_str + "_transform_" \
                         + transform + '_order_is_trainable_' + str(order_is_trainable) + '_order_' + str(order)
    else:
        model_name = "model_" + backbone + "_train_on_" + scene_str + "_transform_" + transform
    model_load_name =os.path.join(model_load_path, model_name)
    print("model load name :", model_load_name)

    if backbone == "resnet18":
        model = ResNet18_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order,
                            only_use_real=only_use_real
                            ).to(device)
    elif backbone== "resnet34":
        model = ResNet34_2D(num_classes=2,
                            num_channels=num_channels,
                            transform=transform,
                            order_is_trainable=order_is_trainable,
                            order=order,
                            only_use_real=only_use_real
                            ).to(device)
    # model = Transformer(1, 4, 4, 4, 256, 2).to(device)
    model.load_state_dict(torch.load(model_load_name))
    
    model.eval()

    with torch.no_grad():
        for test_scene_name in test_scene_names:
            scene_path = os.path.join(test_scene_path, test_scene_name)
            test_loader = get_dataloader(scene_path, batch_size=batch_size, num_channels=num_channels)
            correct, total = 0, 0
            for waveform, labels in test_loader:
                waveform, labels = waveform.to(device), labels.to(device)  # Move to GPU
                # waveform = waveform.permute(0, 2, 1)
                outputs = model(waveform)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Accuracy on {test_scene_name}: {100 * correct / total:.2f}%')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Cross-scene')
    parser.add_argument('--data_path', default='./data/nlos_data_segmented_two_channel/', help='DATA_ROOT')
    parser.add_argument('--train_scene_name', default=['scene2_hall', 'scene3_811'],nargs='+')
    parser.add_argument('--test_scene_name', default=['scene1_corridor'],nargs='+',)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--order_is_trainable', action="store_true")
    parser.add_argument('--only_use_real', action="store_true")
    parser.add_argument('--order', default=0.5, type=float)
    parser.add_argument('--feature_extraction_layer', default='frft')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--model_save_path', default='./checkpoints_revise/', help='Checkpoints_ROOT')
    # args parse
    args = parser.parse_args()
    data_path, train_scene_name, test_scene_name, batch_size, num_channels, only_use_real = args.data_path, args.train_scene_name, \
        args.test_scene_name, args.batch_size, args.num_channels, args.only_use_real
    order_is_trainable, order, feature_extraction_layer, backbone, model_save_path = \
        args.order_is_trainable, args.order, args.feature_extraction_layer, args.backbone, args.model_save_path

    test(test_scene_path=data_path,
         test_scene_names=test_scene_name,
         train_scene_name=train_scene_name,
         batch_size=batch_size,
         num_channels= num_channels,
         order_is_trainable=order_is_trainable,
         order=order,
         transform=feature_extraction_layer,
         backbone=backbone,
         model_load_path=model_save_path,
         only_use_real=only_use_real,
         )
