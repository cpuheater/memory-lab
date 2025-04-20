import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tyro
from dataclasses import dataclass
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from networks.select_model import select_model
from tqdm import tqdm
from datasets.copy_task_dataset import CopyTaskDataset

@dataclass
class Args:
    batch_size: int = 100
    """"""
    epochs: int = 12
    """"""
    lr: int = 0.001
    """"""
    hidden_dim: int = 128
    """"""
    seq_dim: int = 10
    """"""
    input_dim: int = 10
    """"""
    output_dim: int = 10
    """"""
    model_type: str = "CustomLSTM"
    """CustomLSTM | LSTM | sLSTM"""
    blank_length = 200
    """"""
    signal_length = 10


seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)
args = tyro.cli(Args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = f'{os.path.basename(__file__).rstrip(".py")}_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}'
writer = SummaryWriter(f"runs/{exp_name}")

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 2)
    return torch.sum(preds == labels).item()

def save_model(model):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(),
               os.path.join(args.model_dir, f'{exp_name}.pt'))

def train(model, epochs, train_loader, valid_loader, criterion):
    for epoch in range(1, epochs + 1):
        train_loss = 0
        train_accuracy = []
        model.train()
        start = time.time()
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), torch.squeeze(labels).long().to(device)
            outputs = model(images)
            loss = criterion(outputs.transpose(1, 2), labels)
            train_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuracy.append(accuracy(outputs, labels) / (labels.shape[0] * labels.shape[1]))

        mean_train_accuracy = 100 * np.mean(train_accuracy)
        print(f'[{time_since(start)}] Train Epoch: {epoch} Loss: {train_loss} Accuracy: {np.mean(mean_train_accuracy)}')
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", mean_train_accuracy, epoch)
        valid_loss = 0
        valid_accuracy = []
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader, 1):
                images, labels = images.to(device), torch.squeeze(labels).long().to(device)
                outputs = model(images)
                log_probs = F.log_softmax(outputs, dim=2)
                loss = criterion(log_probs.transpose(1, 2), labels)
                valid_loss += loss.item()
                valid_accuracy.append(accuracy(outputs, labels) / (labels.shape[0] * labels.shape[1]))
        mean_valid_accuracy = 100 * np.mean(valid_accuracy)
        writer.add_scalar("valid/loss", valid_loss, epoch)
        writer.add_scalar("valid/accuracy", mean_valid_accuracy, epoch)
        print(f"[{time_since(start)}] Valid Epoch: {epoch} Loss: {valid_loss} Accuracy: {mean_valid_accuracy}")


if __name__ == '__main__':
    train_dataset = CopyTaskDataset(signal_length=args.signal_length, blank_length=args.blank_length)  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = CopyTaskDataset(signal_length=args.signal_length, blank_length=args.blank_length, samples=100)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    model = select_model(args.model_type, input_dim = args.input_dim, hidden_dim = args.hidden_dim, output_dim = args.output_dim, use_embed=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"Training for {args.epochs} epochs")
    train(model, args.epochs, train_loader, valid_loader, criterion)
