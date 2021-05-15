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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

data = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\financial phrasebank\processed_fpbank.csv")

sentences = list(data["sentence"])
labels = list(data["sentiment_id"])
#Sentences we want sentence embeddings for
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']
model_name = "ProsusAI/finbert"
# model_name = "bert-base-uncased"
#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = sentence_embeddings.tolist()

print(numpy.shape(sentence_embeddings))
X_train, X_val, y_train, y_val = train_test_split(sentence_embeddings, labels, test_size=0.2, random_state=42)




start = time.time()

svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train,y_train)

end = time.time()
process = round(end-start,2)
print("Support Vector Machine Classifier has fitted, this process took {} seconds".format(process))

# print(svm_classifier.score(X_val,y_val))
import pickle

#
# Create your model here (same as above)
#

# Save to file in the current working directory
pkl_filename = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\models\SVM_models\svm_bert_finbankphrase_pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_classifier, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

predicted_y = pickle_model.predict(X_val)
print(predicted_y)
predicted_y_proba = pickle_model.predict_proba(X_val)
print(predicted_y_proba)
def compute_metrics(pred,ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)


    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)

    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
    }
    for k in out_dict.keys():
        print(k)
        print(out_dict[k])

compute_metrics(predicted_y,y_val)



#fin
# tensor([[ 0.0102,  0.8853, -0.2012,  ..., -0.6074, -0.5253,  0.0979],
#         [ 0.2248,  0.5273, -0.0640,  ..., -0.1256, -0.5626,  0.1556],
#         [ 0.0105,  0.1267,  0.0584,  ..., -0.2656,  0.5670,  0.4216]])

#base
# tensor([[-0.1256, -0.0235,  0.0972,  ..., -0.1809, -0.3674,  0.2712],
#         [ 0.0789, -0.2891,  0.0363,  ..., -0.1744, -0.4208,  0.6002],
#         [-0.0145, -0.0749,  0.0564,  ..., -0.2625,  0.4954,  0.0740]])



# finbert
# (10561, 768)
# Support Vector Machine Classifier has fitted, this process took 20.61 seconds
# 0.6062470421202082

#bert
# (10561, 768)
# Support Vector Machine Classifier has fitted, this process took 20.34 seconds
# 0.6100331282536677

# just bert
# accuracy
# 0.8843537414965986
# f1
# [0.95479204 0.78947368 0.71287129]
# precision
# [0.92307692 0.78947368 0.87804878]
# recall
# [0.98876404 0.78947368 0.6       ]
# confusiton_mat
# [[264   3   0]
#  [ 19  90   5]
#  [  3  21  36]]


# accuracy
# 0.9841269841269841
# f1
# [0.99625468 0.97835498 0.94017094]
# precision
# [0.99625468 0.96581197 0.96491228]
# recall
# [0.99625468 0.99122807 0.91666667]
# confusiton_mat
# [[266   0   1]
#  [  0 113   1]
#  [  1   4  55]]