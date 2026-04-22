import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

def train(model, dataset_train, dataset_val, epochs=90, lr=0.001, model_name="medium"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)	

    model = model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=4, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lost_hist = [[], []]
    rank_hist = [[], []]

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1} / {epochs}]")

        # train loop
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(dataloader_train, 0):
            feat, labels = data
            feat = feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        lost_hist[0].append(running_loss / len(dataloader_train))
        rank_hist[0].append(correct / total)
        print(f"AVG train loss: {running_loss / len(dataloader_train) }")

        # validation loop
        total, correct, val_loss = test_single_pass(model, dataloader_val, device, criterion)
        lost_hist[1].append(val_loss)
        rank_hist[1].append(correct / total)
        print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%\nValidation loss: {val_loss:.3f}")

    torch.save(model.state_dict(), f"{model_name}_model_weights.pth")
    return lost_hist, rank_hist
        
def test(model, dataset, weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    #dataset_test = StrongPasswordData("test-data.csv", transform)
    dataloader_test = DataLoader(dataset=dataset, batch_size=16, num_workers=4)

    model.load_state_dict(torch.load(weights_file, weights_only=True))
    total, correct, _ = test_single_pass(model, dataloader_test, device)

    print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")
    return correct / total

def test_single_pass(model, dataloader, device, criterion=None):
    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            feat, labels = data
            feat = feat.to(device)
            labels = labels.to(device)

            outputs = model(feat)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total, correct, running_loss / len(dataloader)

# plots an array of losses provided over epochs
def plot_hists(hists, labels, label_end, ylabel, epochs, figname, title):
    x = np.arange(1, epochs + 1)
    for (loss_train, loss_val), label in zip(hists, labels):
        plt.plot(x, loss_train, label=f"{label} train {label_end}")
        plt.plot(x, loss_val, label=f"{label} val {label_end}")

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(figname)
    plt.close()

# plots an array of accuracies provided over epochs
def plot_acc(accs, labels, figname, title):
    x = np.arange(len(accs))
    plt.bar(x, accs)
    for i, v in enumerate(accs):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(figname)
    plt.close()

    return 0;