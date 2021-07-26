import torch.nn as nn


n_katakana = 48 #actual number of characters in the dataset



def generate_model():
    model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Flatten(),
    nn.Dropout(0.1),
    nn.Linear(512,128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128,n_katakana),
    nn.ReLU(),
    nn.Softmax(dim=1),
    )

    return model