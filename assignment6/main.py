import torch
from torchsummary import summary

from model_assignment6 import myCNN

Net = myCNN()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net.to(device)
summary(model, input_size=(3,32,32))