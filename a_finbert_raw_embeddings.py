
from transformers import AutoTokenizer, AutoModel, DistilBertForTokenClassification
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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# #Sentences we want sentence embeddings for
# # sentences = ['This framework generates embeddings for each input sentence',
# #              'Sentences are passed as a list of string.',
# #              'The quick brown fox jumps over the lazy dog.']
# model_name = "ProsusAI/finbert"
# # model_name = "bert-base-uncased"
# #Load AutoModel from huggingface model repository
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
#
# model = DistilBertForTokenClassification.from_pretrained(
#                 "distilbert-base-cased",
#             )
# sentences = ['my token ids here']
# encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
#
# word_embeddings = model.distilbert.embeddings.word_embeddings(encoded_input)
# # word_embeddings_with_positions = model.distilbert.embeddings(["my token ids here"])
#
# print(word_embeddings)



import torch
from transformers import BertModel, BertTokenizer

model_name = "ProsusAI/finbert"
# # model_name = "bert-base-uncased"
# #Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

token_embedding = {token: model.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}

print(len(token_embedding))
# print(token_embedding['development'])
tokens_list = []
embeddings_list = []
for key_v in token_embedding.keys():
    # print(key_v,token_embedding[key_v].tolist())
    tokens_list.append(key_v)
    embeddings_list.append(token_embedding[key_v].tolist())


import pandas as pd

df = pd.DataFrame()
df['token'] = tokens_list
df['embeddings'] = embeddings_list

df.to_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\finbert_raw_embeddings.csv")