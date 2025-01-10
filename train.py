import numpy as np
import json,torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
 
import json

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []
# Recorrer cada frase en los patrones de las intenciones
for intent in intents['intents']:
    tag = intent['tag']
    # agregar a la lista de etiquetas
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokeniza cada palabra de la oración
        w = tokenize(pattern)
        # agregar a la lista de palabras
        all_words.extend(w)
        # agrega una tupla a la lista xy
        xy.append((w, tag))

# Realiza la reducción a su raíz y convierte cada palabra a minúsculas
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Eliminar duplicados y ordenar
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Crear datos de entrenamiento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
     # Lista para almacenar las frases de entrada
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss solo necesita las etiquetas de clase, no el formato one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiperparámetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

   # soporta la indexación de manera que dataset[i] se pueda usar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

   # podemos llamar a len(dataset) para devolver el tamaño
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Entrenar el modelo
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)        
        # Paso hacia adelante
        outputs = model(words)
       # si y fuera one-hot, debemos aplicar
       # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)        
       # Retropropagación y optimización

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
FILE = "chatdata.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')