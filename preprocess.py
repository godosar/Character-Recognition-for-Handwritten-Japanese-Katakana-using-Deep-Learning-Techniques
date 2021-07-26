import torch
from sklearn.model_selection import train_test_split
import numpy as np



n_katakana  = 48 #actual number of characters in the dataset
n_writers = 1411 #number of entries per character
res = 64 #consider a square image



data = torch.load('data/data.pt')



#remove characters 36,38,47 because they are duplicates of 1,3,2 respectively. 48 katakana remain
data = torch.cat([data[:36],data[37:38],data[39:47],data[48:]])



#normalise, flatten, and create labels for the still sorted dataset
data /= torch.max(data)
data = data.reshape((n_katakana*n_writers,1,res,res))
labels = torch.from_numpy(np.repeat(np.arange(n_katakana), n_writers).astype(np.int64))



#split the dataset into a train-, dev-, and test-set
#the sizes are as follows (for each katakana with 1411 entries): train = 900, dev = 100, test = 411
#use stratify to ensure an even distribution of each category
train_x, test_x, train_y, test_y = train_test_split(data, labels, train_size=1000*n_katakana, stratify=labels)
train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, train_size=900*n_katakana, stratify=train_y)



torch.save(train_x,'data/train_x.pt')
torch.save(train_y,'data/train_y.pt')
torch.save(dev_x,'data/dev_x.pt')
torch.save(dev_y,'data/dev_y.pt')
torch.save(test_x,'data/test_x.pt')
torch.save(test_y,'data/test_y.pt')
