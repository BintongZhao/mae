import os
import datetime
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils.wifi_dataset import WiFiDataset
from utils.crop_for_wifi import CropForWiFi
from utils.mae_feature_extractor import MAEFeatureExtractor
from utils.logger import Logger
from models.mlp_classifier import MLPClassifier
from engine_pretrain_classifier import train


def prepare_mae_feature_extractor(model_name, checkpoint_path, device=None):
    mae_feature_extractor = MAEFeatureExtractor(model_name, checkpoint_path, device)
    return mae_feature_extractor


def prepare_mlp_classifier(input_shape, num_classes, hidden_dim=1024, dropout=0.5, device=None):
    return MLPClassifier(input_shape, num_classes, hidden_dim, dropout, device)


def prepare_dataloader(mode='train', batch_size=4, num_workers=4):
    data_path = os.path.join('../', '../', 'data', 'wifi-dataset')
    transform_train = transforms.Compose([
        CropForWiFi(),
        transforms.ToTensor()])
    dataset_train = WiFiDataset(root_dir=data_path, mode=mode, transform=transform_train)
    shuffle = True if mode == 'train' else False
    return DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'output_dir', timestamp)

    logger = Logger(save_path=output_save_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    dataloader_train = prepare_dataloader(mode='train')
    dataloader_val = prepare_dataloader(mode='val')

    mae_model_name = 'mae_vit_base_patch4'
    mae_checkpoint_path = './output_dir/20241012193444/checkpoint-49.pth'
    mae_feature_extractor = prepare_mae_feature_extractor(mae_model_name, mae_checkpoint_path, device)

    input_shape = (128, 768)
    num_classes = 3
    mlp_classifier = prepare_mlp_classifier(input_shape, num_classes, device=device)

    train(mlp_classifier, mae_feature_extractor, dataloader_train, dataloader_val,
          0.0001, 50, 5, [device], output_save_path, logger)


def test():
    mae_model_name = 'mae_vit_base_patch4'
    mae_checkpoint_path = './output_dir/20241012193444/checkpoint-49.pth'
    input_shape = (128, 768)
    num_classes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae_feature_extractor = prepare_mae_feature_extractor(mae_model_name, mae_checkpoint_path, device)
    mlp_classifier = prepare_mlp_classifier(input_shape, num_classes, device=device)
    input_tensor = torch.randn(64, 1, 128, 16).to(device)

    features = mae_feature_extractor.extract(input_tensor)
    mlp_classifier.eval()
    with torch.no_grad():
        outputs = mlp_classifier(features)

    print(f'Extracted features shape: {features.shape}')
    print(f'Output shape: {outputs.shape}')


if __name__ == '__main__':
    main()
