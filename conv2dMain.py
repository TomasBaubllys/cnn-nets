import torch
from conv2dNet import Conv2dNet
from rockPaperScissorsDataset import RockPaperScissorsDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

def train(epochs=90, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)    
    
    conv2dNet = Conv2dNet().to(device)
    #print(conv2dNet)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = RockPaperScissorsDataset("./rockpaperscissors/train", transform)
    dataset_validation = RockPaperScissorsDataset("./rockpaperscissors/validation", transform)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=16, num_workers=4, shuffle=True)
    dataloader_validation = DataLoader(dataset=dataset_validation, batch_size=16, num_workers=4)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(conv2dNet.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1} / {epochs}]")

        # train loop
        conv2dNet.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            imgs, labels = data
            imgs.to(device)
            labels.to(device)

            optimizer.zero_grad()
            outputs = conv2dNet(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"AVG train loss: {running_loss / len(dataloader_train) }")

        # validation loop
        total, correct = test_single_pass(conv2dNet, dataloader_validation, device)
        print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")

    torch.save(conv2dNet.state_dict(), "model_weights.pth")


def test(weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv2dNet = Conv2dNet().to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_test = RockPaperScissorsDataset("./rockpaperscissors/test", transform)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=16, num_workers=4)

    conv2dNet.load_state_dict(torch.load(weights_file, weights_only=True))
    total, correct = test_single_pass(conv2dNet, dataloader_test, device)

    print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")

# does a single pass on a model 
def test_single_pass(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            imgs, labels = data
            imgs.to(device)
            labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total, correct

if __name__ == "__main__":
    #train(10)
    test("model_weights.pth")