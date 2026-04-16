import torch
from conv2dnet.conv2dNet import Conv2dNetStd, Conv2dNetRes, Conv2dNetResBig
from conv2dnet.rockPaperScissorsDataset import RockPaperScissorsDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from utils.nnutils import test, train

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/train", transform)
    dataset_val = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/validation", transform)
    dataset_test = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/test", transform)

    modelSmall = Conv2dNetResBig()
    loss, rank = train(modelSmall, dataset_train, dataset_val, 10, model_name="rps_small")
    print(loss)
    print(rank)
    test(modelSmall, dataset_val, "rps_small_model_weights.pth")