
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

token_embedding = {token: bert.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}

print(len(token_embedding))
print(token_embedding['[CLS]'])