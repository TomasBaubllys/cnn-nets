import torch
from conv2dnet.conv2dNet import Conv2dNetStd, Conv2dNetRes, Conv2dNetResBig
from conv2dnet.rockPaperScissorsDataset import RockPaperScissorsDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from utils.nnutils import test, train, plot_hists, plot_acc, visualize_predictions
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bt", "--base_test", action="store_true", help="Run the test on standart models", default=False)
    parser.add_argument("-btt", "--base_test_train", action="store_true", help="Run the training and testing on standart models", default=False)
    parser.add_argument("-ne", "--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-dpt", "--dropout_test", action="store_true", help="Runs the dropout test on Conv2dResBig", default=False)
    parser.add_argument("-pt", "--pooling_test", action="store_true", default=False, help="Runs the pooling test on Conv2dResBig")
    parser.add_argument("-lrt", "--learning_rate_test", action="store_true", default=False, help="Runs the learning rate test on Conv2dResBig")
    parser.add_argument("-vt", "--visual_test", action="store_true", default=False, help="Visualize the predictions of random 25 images using Conv2dResBig")

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
        test(modelSmall, dataset_test, "defaultNetStd_model_weights.pth")
        test(modelRes, dataset_test, "defaultRes_model_weights.pth")
        test(modelResBig, dataset_test, "defaultResBig_model_weights.pth")
    if args.base_test_train:
        loss1, rank1 = train(modelSmall, dataset_train, dataset_val, args.num_epochs, model_name="defaultNetStd")
        loss2, rank2 = train(modelRes, dataset_train, dataset_val, args.num_epochs, model_name="defaultRes")
        loss3, rank3 = train(modelResBig, dataset_train, dataset_val, args.num_epochs, model_name="defaultResBig")
        plot_hists([loss1, loss2, loss3], ["Std", "Res", "ResBig"], "loss", "Loss", len(loss1[0]), "default_train_loss.jpg", "Base models loss")
        plot_hists([rank1, rank2, rank3], ["Std", "Res", "ResBig"], "rank", "Rank", len(loss1[0]), "default_train_rank.jpg", "Base models rank")


        acc1 = test(modelSmall, dataset_test, "defaultNetStd_model_weights.pth")
        acc2 = test(modelRes, dataset_test, "defaultRes_model_weights.pth")
        acc3 = test(modelResBig, dataset_test, "defaultResBig_model_weights.pth")
        plot_acc([acc1, acc2, acc3], ["Std", "Res", "ResBig"], "default_test_rank.jpg", "Base models test rank")
    if args.dropout_test:
        losss = []
        ranks = []
        accs = []
        dropouts = [0, 0.2, 0.5, 0.7, 0.9]
        test_model = Conv2dNetResBig()
        for dropout in dropouts:
            model = Conv2dNetResBig(dropout=dropout)
            loss, rank = train(model, dataset_train, dataset_val, args.num_epochs, model_name=f"ResBigdp{str(dropout).replace(".", "_")}")
            losss.append(loss)
            ranks.append(rank)
            acc = test(test_model, dataset_test, f"ResBigdp{str(dropout).replace(".", "")}_model_weights.pth")
            accs.append(acc)

        plot_hists(losss, ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "loss", "Loss", len(losss[0][0]), "dpRB_train_loss.jpg", "Dropout significance loss", fancy_legend=True)
        plot_hists(ranks, ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "rank", "Rank", len(losss[0][0]), "dpRB_train_rank.jpg", "Dropout significance rank", fancy_legend=True)
        plot_acc(accs, ["dp=0", "dp=0.2", "dp=0.5", "dp=0.7", "dp=0.9"], "dpRB_test_rank.jpg", "Dropout significance test rank")
    if args.pooling_test:
        pool_arr = [nn.AvgPool2d(kernel_size=2, stride=1, padding=1), 
                    nn.AvgPool2d(kernel_size=4, stride=1, padding=1),
                    nn.AvgPool2d(kernel_size=4, stride=4, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=4, stride=1, padding=1),
                    ]
        labels = ["avg k=2 s=1", "avg k=4 s=1","avg k=4 s=4","max k=2 s=1","max k=4 s=1"]
        losss = []
        ranks = []
        accs = []
        for label, pool in zip(labels, pool_arr):
            model = Conv2dNetResBig(prep_pool=pool)
            loss, rank = train(model, dataset_train, dataset_val, args.num_epochs, model_name=f"ResBigpt{label.replace(" ", "_")}")
            losss.append(loss)
            ranks.append(rank)
            acc = test(model, dataset_test, f"ResBigpt{label.replace(" ", "_")}_model_weights.pth")
            accs.append(acc)

        plot_hists(losss, labels, "loss", "Loss", len(losss[0][0]), "dpPT_train_loss.jpg", "Pooling train loss", fancy_legend=True)
        plot_hists(ranks, labels, "rank", "Rank", len(losss[0][0]), "dpPT_train_rank.jpg", "Pooling train rank", fancy_legend=True)
        plot_acc(accs, labels, "dpPT_test_rank.jpg", "Pooling test rank")
    if args.learning_rate_test:
        lrs = [0.05, 0.01, 0.001, 0.0005, 0.0001]
        losss = []
        ranks = []
        accs = []
        labels = [f"lr = {x}" for x in lrs]
        for lr in lrs:
            model = Conv2dNetResBig()
            loss, rank = train(model, dataset_train, dataset_val, args.num_epochs, model_name=f"ResBiglr{str(lr).replace(".", "_")}", lr=lr)
            losss.append(loss)
            ranks.append(rank)
            acc = test(model, dataset_test, f"ResBiglr{str(lr).replace(".", "_")}_model_weights.pth")
            accs.append(acc)
        
        plot_hists(losss, labels, "loss", "Loss", len(losss[0][0]), "dpLR_train_loss.jpg", "Learning rate train loss", fancy_legend=True)
        plot_hists(ranks, labels, "rank", "Rank", len(losss[0][0]), "dpLR_train_rank.jpg", "Learning rate train rank", fancy_legend=True)
        plot_acc(accs, labels, "dpLR_test_rank.jpg", "Learning rate test rank")
    if args.visual_test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Conv2dNetResBig().to(device)
        train(model, dataset_train, dataset_val, 10)

        #load_info = model.load_state_dict(torch.load("defaultResBig_model_weights.pth", map_location=torch.device('cpu'), weights_only=True))
        #print(load_info)
        visualize_predictions(model, dataset_test, device)



        