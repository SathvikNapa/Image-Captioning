import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,num_layers = 1,bias = True,batch_first = True,dropout = 0,bidirectional = False)
        
        self.linear = nn.Linear(hidden_size, vocab_size) 

    def forward(self, features, captions):
        captions = captions[:, :-1]

        batch_size = features.shape[0]
        self.hidden = self.init_hidden(batch_size)

        embeddings = self.word_embeddings(captions)
            
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim = 1)
            
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
            
        outputs = self.linear(lstm_out)

        return outputs
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
            if (max_indice == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        return output
     
    def get_output(self, inputs, hidden):
        
        lstm_out, hidden = lstm(inputs, hidden)
        outputs = self.linear(lstm_out)
        outputs = outputs.squeeze(1)
                
        return outputs, hidden
                
    def get_next_word_input(self, max_indice):
        inputs = self.word_embeddings(max_indice)
        inputs = inputs.unsqueeze(1)
                
        return inputs        

    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))