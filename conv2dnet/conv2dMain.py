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
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bt", "--base_test", action="store_true", help="Run the test on standart models", default=False)
    parser.add_argument("-btt", "--base_test_train", action="store_true", help="Run the training and testing on standart models", default=False)
    parser.add_argument("-ne", "--num_epochs", type=int, default=10, help="Number of epochs to train for")

    return parser.parse_args()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load the datasets
    dataset_train = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/train", transform)
    dataset_val = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/validation", transform)
    dataset_test = RockPaperScissorsDataset("conv2dnet/rockpaperscissors/test", transform)

    # create all the models
    modelSmall = Conv2dNetStd()
    modelRes = Conv2dNetRes()
    modelResBig = Conv2dNetResBig()

    args = parse_arguments()
    
    if args.base_test:
        test(modelSmall, dataset_val, "defaultNetStd_model_weights.pth")
        test(modelRes, dataset_val, "defaultRes_model_weights.pth")
        test(modelResBig, dataset_val, "defaultResBig_model_weights.pth")
    elif args.base_test_train:
        loss1, rank1 = train(modelSmall, dataset_train, dataset_val, args.num_epochs, model_name="defaultNetStd")
        loss2, rank2 = train(modelRes, dataset_train, dataset_val, args.num_epochs, model_name="defaultRes")
        loss3, rank3 = train(modelResBig, dataset_train, dataset_val, args.num_epochs, model_name="defaultResBig")

        test(modelSmall, dataset_val, "defaultNetStd_model_weights.pth")
        test(modelRes, dataset_val, "defaultRes_model_weights.pth")
        test(modelResBig, dataset_val, "defaultResBig_model_weights.pth")

    modelSmall = Conv2dNetResBig()
    loss, rank = train(modelSmall, dataset_train, dataset_val, 10, model_name="rps_small")
    print(loss)
    print(rank)
    test(modelSmall, dataset_val, "rps_small_model_weights.pth")