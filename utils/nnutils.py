import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

def train(model, dataset_train, dataset_val, epochs=90, lr=0.001, model_name="medium"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=4, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1} / {epochs}]")

        # train loop
        model.train()
        running_loss = 0.0
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

        print(f"AVG train loss: {running_loss / len(dataloader_train) }")

        # validation loop
        total, correct = test_single_pass(model, dataloader_val, device)
        print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")

    torch.save(model.state_dict(), f"{model_name}_model_weights.pth")
        
def test(model, dataset, weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    transform = None

    #dataset_test = StrongPasswordData("test-data.csv", transform)
    dataloader_test = DataLoader(dataset=dataset, batch_size=16, num_workers=4)

    model.load_state_dict(torch.load(weights_file, weights_only=True))
    total, correct = test_single_pass(model, dataloader_test, device)

    print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")

def test_single_pass(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            feat, labels = data
            feat = feat.to(device)
            labels = labels.to(device)

            outputs = model(feat)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total, correct