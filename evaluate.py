import numpy as np
import torch
import model_util
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import imageio



def plot_history(epoch):
    '''
    plot the train- and dev loss over epochs
    '''
    train_losses = np.load(f'models/history_train_loss_{epoch}_epochs.npy', allow_pickle=True)
    dev_losses = np.load(f'models/history_dev_loss_{epoch}_epochs.npy', allow_pickle=True)

    plt.figure(figsize=[10,5])
    epochs = np.arange(0,epoch)
    plt.plot(epochs,train_losses)
    plt.plot(epochs,dev_losses)
    plt.title('training history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train loss','dev loss'])
    plt.savefig('figures/training_history.png')



def predict():
    '''
    evaluate the model on the test set to get the predictions
    '''
    #CUDA compatible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #the scores are calculated incrementally to limit memory usage
    scores = None
    for i in np.linspace(0,len(test_x)-411,48):
        if scores is None:
            scores = model(test_x[:411,:,:,:].to(device)).to('cpu').detach().numpy()
        else:
            scores = np.concatenate([scores,model(test_x[int(i):int(i+411),:,:,:].to(device)).to('cpu').detach().numpy()])
    model.to('cpu')

    predictions = np.argmax(scores,axis=1)
    return predictions



def accuracy():
    truth = test_y.detach().numpy()
    acc = np.sum(predictions == truth)/len(truth)
    return acc



def plot_confusion_matrix():
    truth = test_y.detach().numpy()
    confusion = confusion_matrix(truth,predictions)
    fig = plt.figure(figsize=(18,18))
    ax = plt.axes()
    ConfusionMatrixDisplay(confusion).plot(ax = ax)
    plt.savefig('figures/confusion_matrix.png')



def plot_random_examples(height, width):
    #read random entries
    data = torch.load('data/data.pt')
    data = np.array(data)
    images = []
    for _ in range(height*width):
        images.append(data[np.random.randint(0,48), np.random.randint(0,1411)])
    images = np.array(images)
    print(images.shape)

    #reshape them in such a way, that they form a consistent image
    lines = []
    for i in range(width):
        lines.append(np.concatenate(images[i*height:(i+1)*height]))
        print(lines[i].shape)
    image = np.concatenate(lines,axis=1)
    print(image.shape)
    imageio.imwrite(f'figures/examples.png',image)



def plot_examples(k):
    #produce examples of entries in the dataset
    data = torch.load('data/data.pt')
    if k == 40: #account for the shift in indices from the duplicated categories
        imageio.imwrite(f'figures/examples_38.png',data[k,:4].reshape(4*64,1*64))
    elif k == 49:
        imageio.imwrite(f'figures/examples_46.png',data[k,:4].reshape(4*64,1*64))
    else:
        imageio.imwrite(f'figures/examples_{k}.png',data[k,:4].reshape(4*64,1*64))



test_x = torch.load('data/test_x.pt')
test_y = torch.load('data/test_y.pt')


model = model_util.generate_model()
model.eval()


#the (final) epoch of training, that is the fully trained model is loaded
#in the case of the paper, the training is finished after 250 epochs
epoch = 250
plot_history(epoch)

model.load_state_dict(torch.load(f'models/model_{epoch}_epochs.pt'))
predictions = predict()
acc = accuracy()
print('Accuracy:',acc)

plot_confusion_matrix()
plot_random_examples(4,6)

for i in [16,18,12,22,40,49,4,29,20]:
    plot_examples(i)

print('done')
