import torch
from conv2dnet.conv2dNet import Conv2dNetStd, Conv2dNetRes, Conv2dNetResBig
from conv2dnet.rockPaperScissorsDataset import RockPaperScissorsDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from utils.nnutils import test, train, plot_hists, plot_acc
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bt", "--base_test", action="store_true", help="Run the test on standart models", default=False)
    parser.add_argument("-btt", "--base_test_train", action="store_true", help="Run the training and testing on standart models", default=False)
    parser.add_argument("-ne", "--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-dpt", "--dropout_test", action="store_true", help="Runs the dropout test on Conv2dResBig", default=False)

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
        plot_hists([loss1, loss2, loss3], ["Std", "Res", "ResBig"], "loss", "Loss", len(loss1[0]), "default_train_loss.jpg", "Base models loss")
        plot_hists([rank1, rank2, rank3], ["Std", "Res", "ResBig"], "rank", "Rank", len(loss1[0]), "default_train_rank.jpg", "Base models rank")


        acc1 = test(modelSmall, dataset_val, "defaultNetStd_model_weights.pth")
        acc2 = test(modelRes, dataset_val, "defaultRes_model_weights.pth")
        acc3 = test(modelResBig, dataset_val, "defaultResBig_model_weights.pth")
        plot_acc([acc1, acc2, acc3], ["Std", "Res", "ResBig"], "default_test_rank.jpg", "Base models test rank")
    elif args.dropout_test:
        model0 = Conv2dNetResBig(dropout=0)
        model02 = Conv2dNetResBig(dropout=0.2)
        model05 = Conv2dNetResBig(dropout=0.5)
        model07 = Conv2dNetResBig(dropout=0.7)
        model09 = Conv2dNetResBig(dropout=0.9)
        loss1, rank1 = train(model0, dataset_train, dataset_val, args.num_epochs, model_name="ResBigdp0")
        loss2, rank2 = train(model02, dataset_train, dataset_val, args.num_epochs, model_name="ResBigdp02")
        loss3, rank3 = train(model05, dataset_train, dataset_val, args.num_epochs, model_name="ResBigdp05")
        loss4, rank4 = train(model07, dataset_train, dataset_val, args.num_epochs, model_name="ResBigdp07")
        loss5, rank5 = train(model09, dataset_train, dataset_val, args.num_epochs, model_name="ResBigdp09")
        plot_hists([loss1, loss2, loss3, loss4, loss5], ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "loss", "Loss", len(loss1[0]), "dpRB_train_loss.jpg", "Dropout significance loss")
        plot_hists([rank1, rank2, rank3, loss4, loss5], ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "rank", "Rank", len(loss1[0]), "dpRB_train_rank.jpg", "Dropout significance rank")


        acc1 = test(model0, dataset_val, "ResBigdp0_model_weights.pth")
        acc2 = test(model02, dataset_val, "ResBigdp02_model_weights.pth")
        acc3 = test(model05, dataset_val, "ResBigdp05_model_weights.pth")
        acc4 = test(model07, dataset_val, "ResBigdp07_model_weights.pth")
        acc5 = test(model09, dataset_val, "ResBigdp09_model_weights.pth")
        plot_acc([acc1, acc2, acc3, acc4, acc5], ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "dpRB_test_rank.jpg", "Dropout significance test rank")