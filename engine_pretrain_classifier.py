import os
import torch
from torch import nn
from utils.accumulator import Accumulator


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    cmp = y_hat == y
    return float(cmp.type(y.dtype).sum()) / y.shape[0]


def evaluate(net, feature_extractor, data_iter, loss_func):
    # 验证损失之和, 正确预测的样本数, 样本数
    metric = Accumulator(3)
    device = next(iter(net.parameters())).device
    net.eval()
    with torch.no_grad():
        for x, y in data_iter:
            # x: [batch_size, channel_size, aoa, tof] -> [batch_size, ?, ?]
            x = feature_extractor.extract(x)
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            metric.add(loss_func(y_hat, y).item() * y.shape[0], accuracy(y_hat, y) * y.shape[0], y.shape[0])
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, feature_extractor, train_iter, val_iter, learning_rate, num_epochs, patience, devices, checkpoint_save_dir_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net = net.to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_state_dict = net.state_dict()
    min_val_loss = float('inf')
    min_val_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # 训练损失之和, 正确预测的样本数, 样本数
        metric = Accumulator(3)
        net.train()
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            # x: [batch_size, channel_size, aoa, tof] -> [batch_size, ?, ?]
            x = feature_extractor.extract(x)
            x, y = x.to(devices[0]), y.to(devices[0])
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * y.shape[0], accuracy(y_hat, y) * y.shape[0], y.shape[0])
        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        val_loss, val_acc = evaluate(net, feature_extractor, val_iter, loss_func)
        logger.record(
            f'Epoch: {epoch}, current patience: {current_patience + 1}, train loss: {train_loss:.3f}, train acc: {train_acc:.3f}, val loss: {val_loss:.3f}, val acc: {val_acc:.3f}')
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_save_path = os.path.join(checkpoint_save_dir_path, f"checkpoint-{epoch}.pth")
            torch.save(net.state_dict(), checkpoint_save_path)
        if val_loss < min_val_loss:
            best_state_dict = net.state_dict()
            min_val_loss = val_loss
            min_val_loss_epoch = epoch
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logger.record(f'Early stopping after {epoch + 1} epochs')
                break

    torch.save(best_state_dict, os.path.join(checkpoint_save_dir_path, "best_state_dict.pth"))
    logger.record(f"The best testing loss occurred in the {min_val_loss_epoch} epoch")
