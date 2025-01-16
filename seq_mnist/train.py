import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import math
import torch
import torch.nn as nn
import os
import tyro
from dataclasses import dataclass
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from networks import LSTMWithLinear, CustomLSTMWithLinear, sLSTMWithLinear
from tqdm import tqdm

@dataclass
class Args:
    batch_size: int = 100
    """"""
    epochs: int = 12
    """"""
    lr: int = 0.1
    """"""
    hidden_dim: int = 128
    """"""
    seq_dim: int = 28
    """"""
    input_dim: int = 28
    """"""
    output_dim: int = 10
    """"""
    model_type: str = "sLSTM"
    """CustomLSTM | LSTM | sLSTM"""

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
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item()

def save_model(model):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save(model.state_dict(),
               os.path.join(args.model_dir, f'{exp_name}.pt'))


def train(model, epoch, train_loader, valid_loader, criterion):
    train_loss = 0
    train_accuracy = 0
    model.train()
    start = time.time()
    for i, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.view(-1, args.seq_dim, args.input_dim).to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_accuracy += accuracy(outputs, labels)

    print(f'[{time_since(start)}] Train Epoch: {epoch} Loss: {train_loss} Accuracy: {100 * train_accuracy/len(train_loader.dataset)}')
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/accuracy", 100 * train_accuracy/len(train_loader.dataset), epoch)
    valid_loss = 0
    valid_accuracy = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader, 1):
            images, labels = images.view(-1, args.seq_dim, args.input_dim).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            valid_accuracy += accuracy(outputs, labels)

    writer.add_scalar("valid/loss", valid_loss, epoch)
    writer.add_scalar("valid/accuracy", 100 * valid_accuracy/len(valid_loader.dataset), epoch)
    print(f"[{time_since(start)}] Valid Epoch: {epoch} Loss: {valid_loss} Accuracy: {100 * valid_accuracy/len(valid_loader.dataset)}")


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    def choose_model(model_type: str):
        match model_type:
            case "CustomLSTM":
                return CustomLSTMWithLinear
            case "LSTM":
                return LSTMWithLinear
            case "sLSTM":
                return sLSTMWithLinear

    model = choose_model(args.model_type)(**{"input_dim": args.input_dim, "hidden_dim" : args.hidden_dim, "output_dim": args.output_dim})

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"Training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, train_loader, valid_loader, criterion)
