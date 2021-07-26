import numpy as np
import torch
import time
import model_util



train_x = torch.load('data/train_x.pt')
train_y = torch.load('data/train_y.pt')
dev_x = torch.load('data/dev_x.pt')
dev_y = torch.load('data/dev_y.pt')



#define the model
model = model_util.generate_model()

model.train()

print('number of model parameters:', sum(p.numel() for p in model.parameters()))

torch.save(model.state_dict(),'models/model_0_epochs.pt')
np.save('models/history_train_loss_0_epochs.npy',np.array([]))
np.save('models/history_dev_loss_0_epochs.npy',np.array([]))



#prepare for training and define hyperparameters
optim = torch.optim.Adam(model.parameters(), lr=0.0005)
calculate_loss = torch.nn.CrossEntropyLoss()
dataset = [[train_x[i],train_y[i]] for i in range(len(train_x))]
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)



#this system is designed to be able to dynamically start training from a pretrained model, instead of starting all over again every time, which was helpful during development
#the parameters here are used to reproduce the results presented in the paper
epoch_num = 250
start_epoch = 0

#load the status of the start epoch
#(obsolete when starting with an untrained model)
model.load_state_dict(torch.load(f'models/model_{start_epoch}_epochs.pt'))
train_losses = list(np.load(f'models/history_train_loss_{start_epoch}_epochs.npy', allow_pickle=True))
dev_losses = list(np.load(f'models/history_dev_loss_{start_epoch}_epochs.npy', allow_pickle=True))
print('training from',start_epoch,'for',epoch_num,'epochs')



#the training algorithm is compatible with CUDA for GPU acceleration, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dev_x, dev_y = dev_x.to(device), dev_y.to(device)



def train_one_epoch():
    batch_loss = 0
    for x, y in trainloader:
        #for memory reasons, each batch needs to be transfered to the GPU seperately, so that only one batch is loaded on the GUP at once
        x = x.to(device)
        y = y.to(device)
        loss = calculate_loss(model(x), y)
        loss.backward()
        batch_loss += float(loss) #to reduce memory usage, do not keep the gradients
        optim.step()
        optim.zero_grad()

    #calculate dev loss for validation
    with torch.no_grad():
        batch_loss /= len(trainloader)
        dev_loss = calculate_loss(model(dev_x),dev_y)
    train_losses.append(batch_loss)
    dev_losses.append(dev_loss)

    return batch_loss, dev_loss



toc = time.time()

#the main training algorithm
for e in range(1,epoch_num+1):
    batch_loss, dev_loss = train_one_epoch()
    
    #print the training progress
    print('epoch:',start_epoch+e,'\tloss:',float(batch_loss),'\tdev_loss:',float(dev_loss),end='\r')
    if e % (epoch_num/10) == 0:
        print('epoch:',start_epoch+e,'\tloss:',float(batch_loss),'\tdev_loss:',float(dev_loss),'\ttime:',int((time.time()-toc)/60),'min',int(time.time()-toc)%60,'s')
    
    #save the model and trainig history
    torch.save(model.state_dict(),f'models/model_{start_epoch+e}_epochs.pt')
    np.save(f'models/history_train_loss_{start_epoch+e}_epochs.npy',np.array(train_losses))
    np.save(f'models/history_dev_loss_{start_epoch+e}_epochs.npy',np.array(dev_losses))

print('done!')
