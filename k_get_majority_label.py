import pickle
import pandas as pd
import ast
from datetime import datetime
df = pd.read_csv(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\finbert_raw_embeddings.csv')
# print(df.head())
df = df.dropna()
import numpy as np




def building_embedding_word_list():
    emos = []
    cluster_avg_embeddings = []
    with open(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\financial_emotional_vocabulary.pkl','rb') as f:
        EMO_RESOURCES = pickle.load(f)

    for key in EMO_RESOURCES.keys():
        print(key)
        emos.append(key)
        # print(EMO_RESOURCES[key])

        word_list = EMO_RESOURCES[key]
        tokens = []

        print(word_list)
        vecs = []
        for k in word_list:
            for i, row_e in df.iterrows():
                if (row_e['token'] == k):
                    # print('found')
                    term_vec = row_e['embeddings']
                    print(k, term_vec)
                    tokens.append(k)
                    vecs.append(ast.literal_eval(term_vec))

        labels = [key] * len(tokens)
        dfff = pd.DataFrame()
        dfff['word'] = tokens
        dfff['embedding'] = vecs
        dfff['label'] = labels

        dfff.to_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\term_emb_label\\"+str(key)+".csv")



building_embedding_word_list()