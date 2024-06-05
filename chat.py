import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#check if gpu available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

#callback data
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval() #model evaluate

bot_name = "emmyAi"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('Ask: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) #1 means has one sample and 0 mean num of columns because our model expects it in this shape
    X = torch.from_numpy(X) #convert to torch Enzo because bag of words func return in numpy arrays

    output = model(X) #this will give the prediction
    _, predicted = torch.max(output, dim=1) #dimension = 0
    tag = tags[predicted.item()] #actual tag that we store

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I don't understand...")