# Author: Tomas Baublys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import csv

# generates a confusion matrix for a given model and dataset, before calling ensure that weights file exists
def generate_confusion_matrix(model, dataset, weights_path, save_path="confusion_matrix.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    if hasattr(dataset, 'class_to_idx'):
        class_names = [k for k, v in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    else:
        class_names = [str(i) for i in range(len(np.unique(all_labels)))]

    fig_size = 12 if len(class_names) > 5 else 7
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    disp.plot(
        cmap='Blues', 
        ax=ax, 
        xticks_rotation='vertical' if len(class_names) > 5 else 'horizontal',
        values_format='d'
    )
    
    plt.title(f"Confusion Matrix\nWeights: {weights_path}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
# used for training the given mdoel with the given dataset and given parameters
# outputs the weights into a given file
def train(model, dataset_train, dataset_val, epochs=90, lr=0.001, model_name="medium", optim_str="adam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)	

    model = model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=4, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    if optim_str == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_str == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_str == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

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

# used for testing a given model with the given dataset
def test(model, dataset, weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    #dataset_test = StrongPasswordData("test-data.csv", transform)
    dataloader_test = DataLoader(dataset=dataset, batch_size=16, num_workers=4)

    model.load_state_dict(torch.load(weights_file, weights_only=True))
    total, correct, _ = test_single_pass(model, dataloader_test, device)

    print(f"Correct: {correct} / {total} ===> {correct / total * 100 :.3f}%")
    return correct / total

# runs a single pass of all data in the dataloader, does not change the gradients
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

# plots the history of losses, with the given labels
# used for variuos experiments 
def plot_hists(hists, labels, label_end, ylabel, epochs, figname, title, fancy_legend=False):
    plt.figure(figsize=(10, 6))
    x = np.arange(1, epochs + 1)
    
    for (loss_train, loss_val), label in zip(hists, labels):
        line, = plt.plot(x, loss_train, label=f"{label} train {label_end}")
        plt.plot(x, loss_val, '--', label=f"{label} val {label_end}", color=line.get_color())

    plt.title(title)
    
    if fancy_legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, ncol=3)
    else:
        plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

# plots an array of accuracies provided over epochs
# used for all ranks plotting during all of the experiments
def plot_acc(accs, labels, figname, title):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(accs))
    plt.bar(x, accs)
    for i, v in enumerate(accs):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title(title)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

# function asisted by AI (model used - Gemini)
# function visualized the image classfiers output of random 25 images from the test dataset
def visualize_predictions(model, dataset, device="cpu"):
    model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=25, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    if hasattr(dataset, 'class_to_idx'):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    else:
        idx_to_class = {0: "Paper", 1: "Rock", 2: "Scissors"}

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"P: {idx_to_class[preds[i].item()]}\nA: {idx_to_class[labels[i].item()]}", 
                  color=color, fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig("visual_test.jpg")
    #plt.show()

# used for timeline classifier to export given number of predictions to a file
def export_predictions_to_csv(model, dataset, weights_path, filename="predictions.csv", num_samples=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    data_iter = iter(dataloader)
    features, labels = next(data_iter)
    features, labels = features.to(device), labels.to(device)

    if hasattr(dataset, 'class_to_idx'):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    else:
        idx_to_class = {i: str(i) for i in range(100)} 

    results = []

    with torch.no_grad():
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            
            results.append({
                "Nr": i + 1,
                "Tikra Klase": idx_to_class.get(true_idx, true_idx),
                "Modelio Spejimas": idx_to_class.get(pred_idx, pred_idx),
                "Rezultatas": "TEISINGA" if true_idx == pred_idx else "KLAIDA"
            })

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)