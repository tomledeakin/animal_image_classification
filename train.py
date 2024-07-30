import os.path

import torch
import torch.nn as nn
from datasets import AnimalDataset
from models import AdvancedCNN
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomAffine, ColorJitter
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import shutil
from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings('ignore')

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser("Test arguments")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-2)
    parser.add_argument("--momentum", "-m", type=float, default=0.9)
    parser.add_argument("--data-path", "-d", type=str, default="data/animals")
    parser.add_argument("--log-path", "-p", type=str, default="tensorboard/animals")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="trained_models/animals")
    parser.add_argument("--pretrained-checkpoint-path", "-t", type=str, default=None)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        ToTensor(),
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.1),
            shear=10
        ),
        Resize((args.image_size, args.image_size)),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = AnimalDataset(root=args.data_path, is_train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    val_dataset = AnimalDataset(root=args.data_path, is_train=False, transform=test_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    # model = AdvancedCNN(num_classes=10).to(device)   # Our own model
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # model.fc = nn.Linear(512, 10)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights)
    model.classifier[1] = nn.Linear(1280, 10)
    model.to(device)

    # summary(model, (3, args.image_size, args.image_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.pretrained_checkpoint_path:
        checkpoint = torch.load(args.pretrained_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1
    num_iters = len(train_dataloader)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    writer = SummaryWriter(args.log_path)
    for epoch in range(start_epoch, args.epochs):
        # Training step
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            avg_loss = np.mean(train_loss)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, avg_loss))
            writer.add_scalar("Train/Loss", avg_loss, epoch * num_iters + iter)

        # Validation step
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="yellow")
        all_labels = []
        all_predictions = []
        all_losses = []
        with torch.no_grad():
            # with torch.inference_mode():  # from Pytorch 1.9
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                prediction = model(images)  # prediction's shape: [B, 10]
                loss = criterion(prediction, labels)
                # max_values, max_indices = torch.max(prediction, dim=1)
                predicted_classes = torch.argmax(prediction, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted_classes.tolist())
                all_losses.append(loss.item())
        loss = np.mean(all_losses)
        acc = accuracy_score(all_labels, all_predictions)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar("Val/Loss", loss, epoch)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(10)], epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))



if __name__ == '__main__':
    args = get_args()
    train(args)
