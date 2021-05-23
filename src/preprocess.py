import pandas as pd
import torch
import numpy as np
import json

config = ''
with open('config.json', 'r') as f:
    config = json.load(f)

tag = pd.read_csv("data/tags.csv")
job = pd.read_csv("data/job_tags.csv")
user = pd.read_csv("data/user_tags.csv")
initdata = None
userkeyword = user.merge(tag, on="tagID", how="left").drop_duplicates()
userkeyword = userkeyword.groupby("userID")["keyword"].apply(list).reset_index(name="tags")
userkeyword.rename(columns = {'userID' : 'ID'}, inplace = True)
userkeyword['Cnt'] = userkeyword['tags'].apply(len)

jobkeyword = job.merge(tag, on="tagID", how="left").drop_duplicates()
jobkeyword = jobkeyword.groupby("jobID")["keyword"].apply(list).reset_index(name="tags")
jobkeyword.rename(columns = {'jobID' : 'ID'}, inplace = True)
jobkeyword['Cnt'] = jobkeyword['tags'].apply(len)
allkeyword = userkeyword.append(jobkeyword).reset_index(drop=True)   # userid, tags 또는 jobid, tags로 구성된 데이터셋


def onehot(keyword):
    find = False
    onehot_vec = np.zeros(tag.shape[0])
    assert (tag.index[ tag["keyword"] == keyword] + 1).any(), keyword
    onehot_vec[tag.index[ tag["keyword"] == keyword] ] = 1
    return onehot_vec

class Word2VecDataSet(torch.utils.data.Dataset):
    def __init__(self, data):  # 데이터에 list 형태의 tags 필드가 존재해야 함
        super().__init__()
        self.dataset = []
        maxc = data.Cnt.max()
        for i in range(len(data)):
            keywords = data.tags[i]
            for j in range(maxc):
                idx = j % len(keywords)
                center = onehot(keywords[idx])
                peripheral = np.zeros(tag.shape[0])
                for k in range(len(keywords)):
                    if j == k:
                        continue
                    else:
                        #                        center = onehot(keywords[j])
                        temp = onehot(keywords[k])
                        peripheral += temp
                #                        print("c: ", center, "p: ", peripheral)
                self.dataset.append((center, peripheral))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]

def getAllTagList():
    taglist = tag.keyword
    taglist = taglist.tolist()
    taglist.sort()
    return taglist

def getJobTagList():
    taglist = job.merge(tag, on="tagID", how="left").drop_duplicates()
    taglist = taglist.keyword.drop_duplicates().tolist()
    taglist.sort()
    return taglist

def getUserList():
    return userkeyword

def getJobList():
    return jobkeyword

def getAllList():
    return allkeyword

def getDataset():
    dataset = Word2VecDataSet(jobkeyword)
    return dataset

def getTrainLoader():
    dataset = Word2VecDataSet(jobkeyword)
    return torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True), dataset

if __name__ == "__main__":
    print(getJobIdList())









