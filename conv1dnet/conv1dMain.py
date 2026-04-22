import torch
from conv1dnet.conv1dNet import Conv1dNetSmall, Conv1dNetMedium, Conv1dNetLarge
from conv1dnet.strongPasswordData import StrongPasswordData
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from utils.nnutils import test, train, plot_hists, plot_acc
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bt", "--base_test", action="store_true", help="Run the test on standart models", default=False)
    parser.add_argument("-btt", "--base_test_train", action="store_true", help="Run the training and testing on standart models", default=False)
    parser.add_argument("-ne", "--num_epochs", type=int, default=90, help="Number of epochs to train for")
    parser.add_argument("-bnt", "--batch_norm_test", action="store_true", default=False, help="Run the batch normalization test on Conv1dNetSmall")

    return parser.parse_args()

if __name__ == "__main__":
    dataset_train = StrongPasswordData("conv1dnet/train-data.csv", None)
    dataset_val = StrongPasswordData("conv1dnet/val-data.csv", None)
    dataset_test = StrongPasswordData("conv1dnet/test-data.csv", None)

    args = parse_arguments()

    modelSmall = Conv1dNetSmall()
    modelMedium = Conv1dNetMedium()
    modelLarge = Conv1dNetLarge()

    if args.base_test:
        test(modelSmall, dataset_test, "small_model_weights.pth")
        test(modelMedium, dataset_test, "medium_model_weights.pth")
        test(modelLarge, dataset_test, "large_model_weights.pth")
    if args.base_test_train:
        loss1, rank1 = train(modelLarge, dataset_train, dataset_val, args.num_epochs, model_name="large", lr=3.5e-4)
        loss2, rank2 = train(modelMedium, dataset_train, dataset_val, args.num_epochs, model_name="medium")
        loss3, rank3 = train(modelSmall, dataset_train, dataset_val, args.num_epochs, model_name="small")
        plot_hists([loss1, loss2, loss3], ["Small", "Medium", "Large"], "loss", "Loss", len(loss1[0]), "defaultC1_train_loss.jpg", "Base models loss")
        plot_hists([rank1, rank2, rank3], ["Small", "Medium", "Large"], "rank", "Rank", len(loss1[0]), "defaultC1_train_rank.jpg", "Base models rank")

        acc1 = test(modelSmall, dataset_test, "small_model_weights.pth")
        acc2 = test(modelMedium, dataset_test, "medium_model_weights.pth")
        acc3 = test(modelLarge, dataset_test, "large_model_weights.pth")
        plot_acc([acc1, acc2, acc3], ["Small", "Medium", "Large"], "defaultC1_test_rank.jpg", "Base models test rank")
    if args.batch_norm_test:
        modelSmallBn = Conv1dNetSmall()
        modelSmallNoBn = Conv1dNetSmall(use_batch_norm=False)
        loss1, rank1 = train(modelSmallBn, dataset_train, dataset_val, args.num_epochs, model_name="smallBn")
        loss2, rank2 = train(modelSmallNoBn, dataset_train, dataset_val, args.num_epochs, model_name="smallNoBn")
        plot_hists([loss1, loss2], ["with batchnorm", "without batchnorm"], "loss", "Loss", len(loss1[0]), "smallbn_train_loss.jpg", "Batchnorm train loss")
        plot_hists([rank1, rank2], ["with batchnorm", "without batchnorm"], "rank", "Rank", len(loss1[0]), "smallbn_train_rank.jpg", "Batchnorm train rank")
        acc1 = test(modelSmallBn, dataset_test, "smallBn_model_weights.pth")
        acc2 = test(modelSmallNoBn, dataset_test, "smallNoBn_model_weights.pth")
        plot_acc([acc1, acc2], ["with batchnorm", "without batchnorm"], "smallbn_test_rank.jpg", "Batchnorm test rank")
