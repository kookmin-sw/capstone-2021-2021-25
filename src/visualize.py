import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import engine
import preprocess
config = engine.config
all = preprocess.getAllList()

def sim(keyword1, keyword2):
    k1 = engine.tagembedding(keyword1).reshape(1, -1)
    k2 = engine.tagembedding(keyword2).reshape(1, -1)
    return cosine_similarity(k1, k2)[0][0]

def visualizebyID(user, job, config):
    taglist1 = all[all['ID'] == user].tags.values[0]
    taglist2 = all[all['ID'] == job].tags.values[0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_ticks_position("left")
    ax.set_yticks(range(len(taglist1)))
    ax.set_yticklabels(taglist1)
    ax.scatter(np.zeros(len(taglist1)), np.arange(len(taglist1)))
    ax2 = ax.twinx()
    ax2.set_yticks(range(len(taglist2)))
    ax2.set_yticklabels(taglist2)
    ax2.scatter(np.ones(len(taglist2)), np.arange(len(taglist2)))
    description = "THRESHOLD = " + str(config['THRESHOLD'])
    plt.title(description)
    for i in range(len(taglist1)):
        for j in range(len(taglist2)):
            score = sim(taglist1[i], taglist2[j])
            if score > config['THRESHOLD']:
                linewidth = max((score - config['THRESHOLD']), 0)/config['THRESHOLD']*config['LINEWIDTH']    # 색으로 표현?
                color = plt.get_cmap('Dark2_r')(int(score*100))
                ax.plot([0,1], [i, j*(len(taglist1)-1)/(len(taglist2)-1)], linewidth=linewidth, c=color)

    plt.show()


def visualize_eachtagbyID(user, job, config):
    taglist1 = all[all['ID'] == user].tags.values[0]
    taglist2 = all[all['ID'] == job].tags.values[0]
    description = "THRESHOLD = " + str(config['THRESHOLD'])
    for i in range(len(taglist1)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_ticks_position("left")
        ax.set_yticks(range(len(taglist1)))
        ax.set_yticklabels(taglist1)
        ax.scatter(np.zeros(len(taglist1)), np.arange(len(taglist1)))
        ax2 = ax.twinx()
        ax2.set_yticks(range(len(taglist2)))
        ax2.set_yticklabels(taglist2)
        ax2.scatter(np.ones(len(taglist2)), np.arange(len(taglist2)))
        ax.scatter(0, i, c='r')
        plt.title(description)
        for j in range(len(taglist2)):
            score = sim(taglist1[i], taglist2[j])
            if score > config['THRESHOLD']:
                linewidth = max((score - config['THRESHOLD']), 0)/config['THRESHOLD']*config['LINEWIDTH']    # 색으로 표현?
                color = plt.get_cmap('Dark2_r')(int(score*100))
                ax.plot([0,1], [i, j*(len(taglist1)-1)/(len(taglist2)-1)], linewidth=linewidth, c=color)
        plt.show()

def visualizebyTagList(list1, list2, config):
    taglist1 = list1
    taglist2 = list2
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_ticks_position("left")
    ax.set_yticks(range(len(taglist1)))
    ax.set_yticklabels(taglist1)
    ax.scatter(np.zeros(len(taglist1)), np.arange(len(taglist1)))
    ax2 = ax.twinx()
    ax2.set_yticks(range(len(taglist2)))
    ax2.set_yticklabels(taglist2)
    ax2.scatter(np.ones(len(taglist2)), np.arange(len(taglist2)))
    description = "THRESHOLD = " + str(config['THRESHOLD'])
    plt.title(description)
    for i in range(len(taglist1)):
        for j in range(len(taglist2)):
            score = sim(taglist1[i], taglist2[j])
            if score > config['THRESHOLD']:
                linewidth = max((score - config['THRESHOLD']), 0)/config['THRESHOLD']*config['LINEWIDTH']    # 색으로 표현?
                color = plt.get_cmap('Dark2_r')(int(score*100))
                ax.plot([0,1], [i, j*(len(taglist1)-1)/(len(taglist2)-1)], linewidth=linewidth, c=color)

    plt.show()
def visualize_eachtagbyTagList(list1, list2, config):
    taglist1 = list1
    taglist2 = list2
    description = "THRESHOLD = " + str(config['THRESHOLD'])
    for i in range(len(taglist1)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_ticks_position("left")
        ax.set_yticks(range(len(taglist1)))
        ax.set_yticklabels(taglist1)
        ax.scatter(np.zeros(len(taglist1)), np.arange(len(taglist1)))
        ax2 = ax.twinx()
        ax2.set_yticks(range(len(taglist2)))
        ax2.set_yticklabels(taglist2)
        ax2.scatter(np.ones(len(taglist2)), np.arange(len(taglist2)))
        ax.scatter(0, i, c='r')
        plt.title(description)
        for j in range(len(taglist2)):
            score = sim(taglist1[i], taglist2[j])
            if score > config['THRESHOLD']:
                linewidth = max((score - config['THRESHOLD']), 0)/config['THRESHOLD']*config['LINEWIDTH']    # 색으로 표현?
                color = plt.get_cmap('Dark2_r')(int(score*100))
                ax.plot([0,1], [i, j*(len(taglist1)-1)/(len(taglist2)-1)], linewidth=linewidth, c=color)
        plt.show()

if __name__ == "__main__":
    a = all.ID[0]
    b = all.ID[3]
    visualizebyID(a, b)
    visualizebyTagList(a, b)