import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
from kor2vec import Kor2Vec
from collections import Counter
from sklearn.metrics import confusion_matrix

# custom modules
from loader import *
from model import *
from config import *
from earlystopping import *
from data import *


def inference(input_path, model_name, output_path):
    # kor2vec model load
    word_emb = Kor2Vec.load('kor2vec03010231.checkpoint.ep0')
    loader = LOADER(word_emb)

    # Checkpoint model load
    checkpoint_model = GRU_ATT()
    model = GRU_ATT_WRAP(checkpoint_model)

    checkpoint = torch.load(model_name)
    checkpoint_model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("Checkpoint => Epoch: " + str(epoch) + "  Loss: " + str(round(loss, 6)))

    # data load
    with open(input_path, 'rt', encoding='utf8') as f:
        data = f.readlines()
    data = [item.strip().split('\t') for item in data]
    sample = np.array(data)[:, -1:]
    sampleT = [[loader.tensorize.embedding(item[0])] for item in sample]  # into torch.tensor
    sampleT = [(loader.trim(item[0])) if item[0].size(0) > MAX_SEQ_LEN else (loader.zero_pad(item[0])) for item in
               sampleT]  # if seq len is too long, trim it. otherwise, pad the sequence.
    test = DataLoader(sampleT, batch_size=BATCH_SIZE)

    # Test
    model.model.eval()
    pred = []
    for batch in test:
        out = model.model(batch)
        pred.append(out.max(dim=1)[1])

    relevant_cols = ['정의', '과정', '성질', '예', '흥미유발']
    label_map = dict(zip(range(5), relevant_cols))
    knowledge = []
    index = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            knowledge.append([])
            knowledge[index].append(label_map[int(pred[i][j])])
            index += 1

    # Save
    knowledge_list = np.hstack((sample, knowledge))
    write(output_path, knowledge_list)

def main():
    # model name
    name = "./Classification_2020-03-25.pth"

    inference('6_final_2.tsv', name, "test.tsv")

if __name__ == '__main__':
	main()