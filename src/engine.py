import word2vecmodel
import preprocess
import torch
import torch.nn as nn
import datetime

import json
config = ''
with open('config.json', 'r') as f:
    config = json.load(f)

enc = word2vecmodel.Encoder(config)
dec = word2vecmodel.Decoder(config)
net = word2vecmodel.word2vec(enc, dec)

def tagembedding(keyword):
    inputvec = torch.from_numpy(preprocess.onehot(keyword))
    with torch.no_grad():
        embeddingvec = enc(inputvec.float())
    return embeddingvec

def use_optimizer(model, params):
    if params['OPTIM'] == 'adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=params['LEARNING_RATE'])
    elif params['optimizer'] == 'rmsprop':
        opt = torch.optim.RMSprop(model.parameters(),
                                        lr=params['LEARNING_RATE'],
                                        momentum=params['MOMENTUM'])
    return opt
def train(trainloader):
    model = net
    epoch = config['EPOCH']
    criterion = nn.BCELoss()
    optim = use_optimizer(model, config)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for eps in range(epoch):
        print('training', eps + 1, 'epoch')
        for i, data in enumerate(trainloader, 0):
            model.train()
            optim.zero_grad()
            input, label = data
            label = label.float()
            predict = model(input.float())
            weight = torch.tensor([0.01, 0.99])
            weight_ = weight[label.data.view(-1).long()].view_as(label)
            loss = criterion(predict, label)
            loss = loss * weight_
            loss = loss.mean()
            loss.backward()
            optim.step()
            print("eps: ", eps, "data: ", i)
def save(path):
    torch.save(enc.state_dict(), path)
def load(path):
    enc.load_state_dict(torch.load(path))
    return enc

if __name__ == "__main__":
    train_loader = preprocess.getTrainLoader(config)
    train(train_loader)


















