import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
         # Capa lineal de entrada a la capa oculta
        self.l1 = nn.Linear(input_size, hidden_size) 
        # Capa lineal de la capa oculta a otra capa oculta
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        # Capa lineal de la capa oculta a la capa de salida
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Función de activación ReLU
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out