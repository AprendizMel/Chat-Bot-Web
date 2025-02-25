import os,random,json,torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import torch
data = torch.load('chatdata.pth')
print(data.keys())





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

data_dir = os.path.join(os.path.dirname(__file__))
FILE = os.path.join(data_dir, 'chatdata.pth')

data = torch.load('chatdata.pth', weights_only=True)



input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Enseña por Colombia"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "No entiendo esta pregunta..."
if __name__ == "__main__":
    print("¡Vamos a chatear! (escribe 'quit' para salir)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        resp = get_response(sentence)
        print(resp)
