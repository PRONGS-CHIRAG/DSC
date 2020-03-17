import numpy as np 
import torch as t 
import helper as h 
import matplotlib.pyplot as plt 
from torchvision import datasets,transforms
transform =transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=datasets.MNIST('MNIST_data/',download=True,train=True,transform=transform)
trainloader=t.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
testset=datasets.MNIST('MNIST_data/',download=True,train=False,transform=transform)
testloader=t.utils.data.DataLoader(testset,batch_size=64,shuffle=True)
dataiter=iter(trainloader)
images,labels=dataiter.next()
plt.imshow(images[1].numpy().squeeze(),cmap='Greys_r')
plt.show()