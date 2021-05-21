from transformers import AutoTokenizer, AutoModel
import torch
import numpy
import time
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, \
    confusion_matrix
import torch
import numpy as np
import pandas as pd

import ast
from datetime import datetime
df = pd.read_csv(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\finbert_raw_embeddings.csv')
# print(df.head())
df = df.dropna()
import numpy as np
import pickle
# from scipy import spatial
#
# dataSetI = [1, 0, -1]
# dataSetII = [1,-1, 0]
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
# print(result)
from sklearn.metrics.pairwise import cosine_similarity
dis = cosine_similarity([[1, 0, -1]], [[1,-1, 0]])
# print(dis)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


clusters = {}
emb_emo = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\emotion_cluster_embeddings.csv")
for i,e_row in emb_emo.iterrows():
    # print(e_row)
    clusters[e_row['emotion']] = ast.literal_eval(e_row['avg_cluster_embeddin'])

def compute_label(input_vec):

    for c_ in clusters:
        c_vec = clusters[c_]
        # print(numpy.shape(input_vec[0]))
        # print(numpy.shape(c_vec))
        # print(input_vec[0])
        # print(c_vec)
        c_sim = cosine_similarity([c_vec],[input_vec[0]])
        print(c_,c_sim)

model_name = "ProsusAI/finbert"
# model_name = "bert-base-uncased"
#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
sentences = ['The world is experiencing unprecedenting challenges in economy from COVID-19.']
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = sentence_embeddings.tolist()


print(sentence_embeddings)
print(numpy.shape(sentence_embeddings))
compute_label(sentence_embeddings)