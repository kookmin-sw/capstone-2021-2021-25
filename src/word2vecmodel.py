import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inputdim = config['INPUT_DIM']
        self.hiddendim = config['ENC_HID_DIM']
        self.embeddim = config['EMB_DIM']
        #        self.inputdim = inputdim
        #        self.hiddendim = hiddendim
        #        self.embeddim = embeddim
        self.hidden = nn.Linear(self.inputdim, self.hiddendim)
        self.embed = nn.Linear(self.hiddendim, self.embeddim)

    def forward(self, input):
        embedvec = self.hidden(input)
        embedvec = self.embed(embedvec)
        embedvec = F.softmax(embedvec, dim=0)
        return embedvec


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inputdim = config['INPUT_DIM']
        self.hiddendim = config['DEC_HID_DIM']
        self.embeddim = config['EMB_DIM']

        self.hidden = nn.Linear(self.embeddim, self.hiddendim)
        self.output = nn.Linear(self.hiddendim, self.inputdim)

    def forward(self, input):
        outvec = self.hidden(input)
        outvec = self.output(outvec)
        outvec = F.softmax(outvec, dim=0)
        return outvec


class word2vec(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        #        with torch.no_grad:
        #            center = x[0]
        #            peripheral = x[1]
        #            center = onehot(center)
        #            peripheral = peripheral(center)
        center = x
        center = self.encoder(center)
        predict = self.decoder(center)
        return predict
