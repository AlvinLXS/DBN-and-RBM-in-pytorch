
# coding: utf-8

# In[1]:


# Importing the libraries
from RBM import RBM
import torch 
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader

import matplotlib
import matplotlib.pyplot as plt

import math
import numpy as np


# In[2]:


#Loading MNIST dataset
mnist_data = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose(
                    [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))


# In[3]:


# Need to convert th data into binary variables
mnist_data.train_data = (mnist_data.train_data.type(torch.FloatTensor)/255).bernoulli()


# In[4]:


#Lets us visualize a number from the data set
idx = 5
img = mnist_data.train_data[idx]
print("The number shown is the number: {}".format(mnist_data.train_labels[idx]) )
plt.imshow(img , cmap = 'gray')
plt.show()


# In[5]:


# If we train on the whole set we expect it to learn to detect edges.
batch_size= 512*4
tensor_x = mnist_data.train_data.type(torch.FloatTensor) # transform to torch tensors
tensor_y = mnist_data.train_labels.type(torch.FloatTensor)
_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = torch.utils.data.DataLoader(_dataset,
                    batch_size=batch_size, shuffle=True,drop_last = True)


# In[6]:


# I have have set these hyper parameters although you can experiment with them to find better hyperparameters.
visible_units=28*28
hidden_units = 1024
k=3
learning_rate=0.01
learning_rate_decay = True
xavier_init = True
increase_to_cd_k = False
use_gpu = True


rbm_mnist = RBM(visible_units,hidden_units,k ,learning_rate,learning_rate_decay,xavier_init,
                increase_to_cd_k,use_gpu).cuda()


# In[7]:



epochs = 3

rbm_mnist.train(train_loader , epochs,batch_size)


# In[8]:


# learned_weights = rbm_mnist.W.transpose(0,1).numpy()
# plt.show()
# fig = plt.figure(3, figsize=(10,10))
# for i in range(25):
#     sub = fig.add_subplot(5, 5, i+1)
#     sub.imshow(learned_weights[i,:].reshape((28,28)), cmap=plt.cm.gray)
# plt.show()


# In[9]:


#This is an unsupervised learning algorithm. So let us try training on one particular number.But first
# we need to seperate the data.

number = 5 #A number between 0 and 10.

particular_mnist = []

limit = mnist_data.train_data.shape[0]
# limit = 60000
for i in range(limit):
    if(mnist_data.train_labels[i] == number):
        particular_mnist.append(mnist_data.train_data[i].numpy())
# particular_mnist = np.array(particular_mnist)
len(particular_mnist)
# mnist_data.train_data


# In[10]:


tensor_x = torch.stack([torch.Tensor(i) for i in particular_mnist]).type(torch.FloatTensor)
tensor_y = torch.stack([torch.Tensor(number) for i in range(len(particular_mnist))]).type(torch.FloatTensor)


# In[11]:


mnist_particular_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
mnist_particular_dataloader = torch.utils.data.DataLoader(mnist_particular_dataset,batch_size = batch_size,drop_last=True,num_workers=0)


# In[12]:


visible_units=28*28
hidden_units = 500
k=3
learning_rate=0.01
learning_rate_decay = False
xavier_init = True
increase_to_cd_k = False
use_gpu = True


rbm_mnist = RBM(visible_units,hidden_units,k ,learning_rate,learning_rate_decay,xavier_init,
                increase_to_cd_k,use_gpu)


epochs = 3

rbm_mnist.train(mnist_particular_dataloader , epochs)


# In[13]:


# This shows the weights for each of the 64 hidden neurons and give an idea how each neuron is activated.

# learned_weights = rbm_mnist.W.transpose(0,1).numpy()
# plt.show()
# fig = plt.figure(3, figsize=(10,10))
# for i in range(25):
#     sub = fig.add_subplot(5, 5, i+1)
#     sub.imshow(learned_weights[i, :].reshape((28,28)), cmap=plt.cm.gray)
# plt.show()


# In[14]:


#Lets try reconstructing a random number from this model which has learned 5
idx = 7
img = mnist_data.train_data[idx]
reconstructed_img = img.view(-1).type(torch.FloatTensor)

# _ , reconstructed_img = rbm_mnist.to_hidden(reconstructed_img)
# _ , reconstructed_img = rbm_mnist.to_visible(reconstructed_img)

_,reconstructed_img = rbm_mnist.reconstruct(reconstructed_img,1)
# print(reconstructed_img)
reconstructed_img = reconstructed_img.view((28,28))
print("The original number: {}".format(mnist_data.train_labels[idx]))
plt.imshow(img , cmap = 'gray')
plt.show()
print("The reconstructed image")
plt.imshow(reconstructed_img , cmap = 'gray')
plt.show()

