import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNetStd(nn.Module):
	def __init__(self):
		super().__init__()
		self.cnn_layers = nn.Sequential (
			nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
			nn.AvgPool2d(kernel_size=4, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
			nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
			nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=0),
			nn.BatchNorm2d(4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.fc = nn.Sequential(
			nn.Linear(676, 3)
			#nn.Linear(50176, 3)
		)

	def forward(self, x):
		x = self.cnn_layers(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x
	
class ResBlock(nn.Module):
	def __init__(self, channels=4):
		super().__init__()

		self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)

		self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = F.relu(out, inplace=True)

		out = self.conv2(out)
		out = self.bn2(out)

		out = out + identity
		out = F.relu(out, inplace=True)

		return out

class Conv2dNetRes(nn.Module):
	def __init__(self):
		super().__init__()
		self.res_block = ResBlock(16)
		self.prep_layer = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
			nn.AvgPool2d(kernel_size=4, stride=1, padding=1),
			nn.ReLU(inplace=True)
		)
		self.last_layer = nn.AdaptiveAvgPool2d(4)

		self.fc = nn.Sequential(
			nn.Linear(256, 3)
			#nn.Linear(50176, 3)
		)

	def forward(self, x):
		x = self.prep_layer(x)
		x = self.res_block(x)
		x = self.last_layer(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

class Conv2dNetResBig(nn.Module):
	def __init__(self, dropout=0, prep_pool=nn.AvgPool2d(kernel_size=4, stride=1, padding=1)):
		super().__init__()

		self.dropout = dropout
		self.res_block1 = ResBlock(8)
		self.prep_layer = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
			nn.AvgPool2d(kernel_size=4, stride=1, padding=1),
			nn.ReLU(inplace=True)
		)
		
		self.middle_layer = nn.Sequential(
			nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
			nn.AvgPool2d(4),
			nn.ReLU(inplace=True)
		)

		self.res_block2 = ResBlock(16)

		self.last_layer = nn.AdaptiveAvgPool2d(4)
		
		self.fc = nn.Sequential(
			nn.Dropout(self.dropout),
			nn.Linear(256, 3)
		)

	def forward(self, x):
		x = self.prep_layer(x)
		x = self.res_block1(x)
		x = self.middle_layer(x)
		x = self.res_block2(x)
		x = self.last_layer(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x


if __name__ == "__main__":
	model = Conv2dNetStd()
	print(model)
	
	dummy_input = torch.rand(1, 3, 224, 224)
	print(dummy_input)
	print(model(dummy_input))