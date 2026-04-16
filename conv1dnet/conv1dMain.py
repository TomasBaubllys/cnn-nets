import torch
from conv1dnet.conv1dNet import Conv1dNetSmall, Conv1dNetMedium, Conv1dNetLarge
from conv1dnet.strongPasswordData import StrongPasswordData
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from utils.nnutils import test, train



if __name__ == "__main__":
    dataset_train = StrongPasswordData("conv1dnet/train-data.csv", None)
    dataset_val = StrongPasswordData("conv1dnet/val-data.csv", None)
    dataset_test = StrongPasswordData("conv1dnet/test-data.csv", None)

    #modelLarge = Conv1dNetLarge()
    #train(modelLarge, dataset_train, dataset_val, 90, model_name="large", lr=3.5e-4)
    #test(modelLarge, dataset_test, "large_model_weights.pth")

    #modelMedium = Conv1dNetMedium()
    #train(modelMedium, dataset_train, dataset_val, 90, model_name="medium")
    #test(modelMedium, dataset_test, "medium_model_weights.pth")

    modelSmall = Conv1dNetSmall()
    train(modelSmall, dataset_train, dataset_val, 90, model_name="small")
    test(modelSmall, dataset_test, "small_model_weights.pth")
