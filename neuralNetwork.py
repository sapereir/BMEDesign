from torch import nn

#pulled from: https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
#part 2: https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        #inputs = 10, outputs = 2
        self.hidden = nn.Linear(10, 2)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
        
