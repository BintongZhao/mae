import os
import datetime
import torch

from torch.utils.data import DataLoader
from util.env_dataset import EnvDataset
from util.logger import Logger
from models.unet import UNet
from engine_pretrain_extractor import train


def prepare_dataloader(root_dir, batch_size, mode='train', transform=None, shuffle=True, num_workers=4):
    dataset = EnvDataset(root_dir=root_dir, mode=mode, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'output_dir', timestamp)
    root_dir, batch_size = '/home/xinglibao/WorkSpace/datasets/env', 16

    logger = Logger(save_path=output_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_extractor = UNet(bilinear=False, base_c=32)

    dataloader_train = prepare_dataloader(root_dir, batch_size, mode='train', shuffle=True)
    dataloader_val = prepare_dataloader(root_dir, batch_size, mode='val', shuffle=False)
    dataloader_test = prepare_dataloader(root_dir, batch_size, mode='test', shuffle=False)

    train(unet_extractor, dataloader_train, dataloader_val,
          0.75, 0.0001, 500, 5, [device], output_save_path, logger)


if __name__ == '__main__':
    main()
