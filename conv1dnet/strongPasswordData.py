import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import random

class StrongPasswordData(Dataset):
    def __init__(self, file_path, transform):
        self.transform = transform
        self.file_path = os.path.join(os.getcwd(), file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File: {self.file_path} not found")
        
        with open(self.file_path, "r") as file:
            self.data = file.readlines()
        
        self.data = [[float(num) for num in el.strip('\n').split(',')] for el in self.data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data[index][0]
        feat = self.data[index][1:]

        if self.transform:
            feat = self.transform(feat)

        label = torch.tensor(label, dtype=torch.long)
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        return feat, label

if __name__ == "__main__":
    dataset = StrongPasswordData("test-data.csv", None)
    print(dataset.__len__())
    print(dataset.__getitem__(7))