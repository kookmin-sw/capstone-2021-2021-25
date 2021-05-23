import word2vecmodel
import preprocess
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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

def get_cosim(tag1, tag2): # 두 tag의 코사인 유사도 비교
    tag1 = tag1.reshape(1, -1)
    tag2 = tag2.reshape(1, -1)

    return cosine_similarity(tag1, tag2)[0][0]

def cal_score(user_taglist, joblist, top_k): # 태그 유사도를 기반으로 채용 공고 추천
    job_scores = []
    jobidlist = joblist["ID"].tolist()

    for jobid in jobidlist:  # job 하나
        job_taglist = joblist[joblist['ID'] == jobid]['tags'].tolist()[0]
        tag_scores = dict()

        for job_tag in job_taglist:  # job의 tag 하나
            if job_tag in user_taglist:
                tag_scores[job_tag] = 1
            else:
                tag_scores[job_tag] = 0
                embedded_jobtag = tagembedding(job_tag)

                for user_tag in user_taglist:  # user의 tag 하나
                    embedded_usertag = tagembedding(user_tag)
                    cosim = get_cosim(embedded_jobtag, embedded_usertag)

                    tag_scores[job_tag] = max(cosim * cosim * cosim * cosim, tag_scores[job_tag])

        sum_scores = 0
        for k, v in tag_scores.items():
            sum_scores += v

        job_scores.append(sum_scores / len(job_taglist))

    job_score_pd = pd.DataFrame({'jobID': jobidlist,
                                 'tags': joblist["tags"].tolist(),
                                 'score': job_scores})

    job_score_pd['len'] = job_score_pd['tags'].str.len()
    job_score_pd = job_score_pd.sort_values(by=['score', 'len'], axis=0, ascending=False).drop(columns='len')
    job_score_pd.reset_index(drop=True, inplace=True)

    return job_score_pd[:top_k]

if __name__ == "__main__":
    train_loader = preprocess.getTrainLoader(config)
    train(train_loader)


















