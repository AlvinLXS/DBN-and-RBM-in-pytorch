
# coding: utf-8

# In[1]:


# Importing the libraries
from DBN import DBN
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


mnist_data.train_data = (mnist_data.train_data.type(torch.cuda.FloatTensor)/255).bernoulli()


# In[4]:


#Lets us visualize a number from the data set
idx = 5
img = mnist_data.train_data[idx]
print("The number shown is the number: {}".format(mnist_data.train_labels[idx]) )
plt.imshow(img , cmap = 'gray')
plt.show()


# In[5]:


# I have have set these hyper parameters although you can experiment with them to find better hyperparameters.
dbn_mnist = DBN(visible_units=28*28 ,
                hidden_units=[23*23 ,18*18] ,
                k = 5,
                learning_rate = 0.01,
                learning_rate_decay = True,
                xavier_init = True,
                increase_to_cd_k = False,
                use_gpu = True)


# In[6]:


num_epochs = 2
batch_size = 1000
tensor_x = mnist_data.train_data.cuda()
tensor_y = mnist_data.train_labels.cuda()
dbn_mnist.train_static(tensor_x,tensor_y,num_epochs , batch_size)


# In[ ]:


# visualising layer 1
# learned_weights = dbn_mnist.rbm_layers[0].W.transpose(0,1).numpy()
# plt.show()
# fig = plt.figure(3, figsize=(10,10))
# for i in range(25):
#     sub = fig.add_subplot(5, 5, i+1)
#     sub.imshow(learned_weights[i,:].reshape((28,28)), cmap=plt.cm.gray)
# plt.show()


# In[ ]:


# visualising layer 2
# learned_weights = dbn_mnist.rbm_layers[1].W.transpose(0,1).numpy()
# plt.show()
# fig = plt.figure(3, figsize=(10,10))
# for i in range(25):
#     sub = fig.add_subplot(5, 5, i+1)
#     sub.imshow(learned_weights[i,:].reshape((23,23)), cmap=plt.cm.gray)
# plt.show()


# In[ ]:


number = 5 #A number between 0 and 10.

particular_mnist = []

limit = mnist_data.train_data.shape[0]
# limit = 60000
for i in range(limit):
    if(mnist_data.train_labels[i] == number):
        particular_mnist.append(mnist_data.train_data[i].cpu().numpy())
# particular_mnist = np.array(particular_mnist)
len(particular_mnist)
# mnist_data.train_data


# In[ ]:


train_data = torch.stack([torch.Tensor(i) for i in particular_mnist])
train_label = torch.stack([torch.Tensor(number) for i in range(len(particular_mnist))])


# In[ ]:


dbn_mnist.train_static(train_data,train_label,num_epochs , batch_size)


# In[ ]:


idx = 3
img = mnist_data.train_data[idx]
reconstructed_img = img.view(1,-1).type(torch.FloatTensor)

_,reconstructed_img= dbn_mnist.reconstruct(reconstructed_img)

reconstructed_img = reconstructed_img.view((28,28))
print("The original number: {}".format(mnist_data.train_labels[idx]))
plt.imshow(img , cmap = 'gray')
plt.show()
print("The reconstructed image")
plt.imshow(reconstructed_img , cmap = 'gray')
plt.show()

