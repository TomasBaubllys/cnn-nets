import torch
import torch.nn as nn

class Conv2dNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.cnn_layers = nn.Sequential (
			nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
			nn.AvgPool2d(kernel_size=4, stride=1, padding=1),
			nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.fc = nn.Sequential(
			nn.Linear(12544, 3)
			#nn.Linear(50176, 3)
		)

	def forward(self, x):
		x = self.cnn_layers(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

if __name__ == "__main__":
	model = Conv2dNet()
	print(model)
	
	dummy_input = torch.rand(1, 3, 224, 224)
	print(dummy_input)
	print(model(dummy_input))