import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import random
from PIL import Image

class RockPaperScissorsDataset(Dataset):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.data_path = data_path
        self.imgs_and_labels = []
        self.class_to_idx = {"scissors" : 0, "rock" : 1, "paper" : 2}

        scissors_path = os.path.join(os.getcwd(), data_path, "scissors")
        if not os.path.exists(scissors_path):
            raise FileNotFoundError(f"Directory {scissors_path} does not exist")
        
        rocks_path = os.path.join(os.getcwd(), data_path, "rock")
        if not os.path.exists(rocks_path):
            raise FileNotFoundError(f"Directory {rocks_path} does not exist")
        
        papers_path = os.path.join(os.getcwd(), data_path, "paper")
        if not os.path.exists(papers_path):
            raise FileNotFoundError(f"Directory {papers_path} does not exist")

        scissors = os.listdir(scissors_path)
        for scss in scissors:
            self.imgs_and_labels.append((os.path.join(scissors_path, scss), self.class_to_idx["scissors"]))

        rocks = os.listdir(rocks_path)        
        for rock in rocks:
            self.imgs_and_labels.append((os.path.join(rocks_path, rock), self.class_to_idx["rock"]))
        
        papers = os.listdir(papers_path)
        for paper in papers:
            self.imgs_and_labels.append((os.path.join(papers_path, paper), self.class_to_idx["paper"]))

        random.shuffle(self.imgs_and_labels)
        #print(self.imgs_and_labels)

    def __len__(self):
        return len(self.imgs_and_labels)

    def __getitem__(self, index):
        img = Image.open(self.imgs_and_labels[index][0]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.imgs_and_labels[index][1]
    

if __name__ == "__main__":
    dataset = RockPaperScissorsDataset("./rockpaperscissors/train", transform=None)
    print(dataset.__len__())
    img, label = dataset.__getitem__(10)
    img.show()
    print(label)