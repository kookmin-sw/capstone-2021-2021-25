import word2vecmodel
import preprocess
import torch
import torch.nn as nn
import datetime
config = { 'INPUT_DIM' : 887,
           'EMB_DIM'  : 50,
           'ENC_HID_DIM' : 100,
           'DEC_HID_DIM' : 100,
           'BATCH_SIZE' : 10,
           'EPOCH' : 5,
           'OPTIM' : 'adam',
           'LEARNING_RATE' : 0.001,
           'MOMENTUM' : 0
}

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
def train(model, trainloader):
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
            loss.backward();
            optim.step()
            print("eps: ", eps, "data: ", i)
def save(path):
    torch.save(enc.state_dict(), path)
def load(path):
    word2vecmodel.load_state_dict(torch.load(path))

if __name__ == "__main__":
    train_loader = preprocess.getTrainLoader(config)
    train(word2vecmodel, train_loader)


















