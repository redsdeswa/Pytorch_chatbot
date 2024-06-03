import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)https://github.com/redsdeswa/Pytorch_chatbot/security
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print(all_words)
#print(tags)

X_train = [] #X_train represents the features or inputs for training the chatbot.
Y_train = [] #Y_train represents the labels or targets for training the chatbot.
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label) #CrossEntropyLoss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.Y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.Y_data[index]
    
    def __len__(self):
        return self.n_samples

#hyperparameter  
batch_size = 8
hidden_size = 8
output_size = len(tags) #num of different class or text we have
input_size = len(X_train[0])
#print(input_size, len(all_words))
#print(output_size, tags)
learning_rate = 0.001
num_epoch = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #เช็คว่า cpu รองรับไหม
model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#actual training loop
for epoch in range(num_epoch):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epoch}, loss={loss.item():.4f}')

labels = label.long()
Y_train = torch.tensor(Y_train, dtype=torch.long)
print(Y_train.dtype)
#print('final loss, loss={loss.item{}:.4f}')
