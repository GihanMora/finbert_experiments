
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertModel, BertForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",output_hidden_states=True)
model = BertModel.from_pretrained("ProsusAI/finbert",output_hidden_states=True)
sentence = "good good very nice"
tokens = tokenizer.tokenize(sentence)
print(tokens)

input_ids = torch.tensor(tokenizer.encode(sentence))[None, :]  # Batch size 1

outputs = model(input_ids)
last_hidden_states = outputs  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states)

# #concatenating last four hidden layers
hidden_states = outputs['hidden_states']
# pooled_output = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]))
pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
print('this',pooled_output)
# # print('that',numpy.shape(average_last_4_hidden_states))
# # https://github.com/huggingface/transformers/issues/1328
