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

def compute_label(input_vec):
    clusters = {}
    emb_emo = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\emotion_cluster_embeddings.csv")
    for i,e_row in emb_emo.iterrows():
        print(e_row)
        clusters[e_row['emotion']] = ast.literal_eval(e_row['avg_cluster_embeddin'])


    for c_ in clusters:
        c_vec = clusters[c_]
        c_sim = cosine_similarity([c_vec],[input_vec])
        print(c_,c_sim)



def compute_avg_emb(word_list):
    vecs = []
    for k in word_list:
        for i, row_e in df.iterrows():
            if (row_e['token'] == k):
                # print('found')
                term_vec = row_e['embeddings']
                print(k,term_vec)

                vecs.append(ast.literal_eval(term_vec))


    avg_emb = np.mean(vecs, axis=0)
    print('ok')
    print(list(avg_emb))
    return list(avg_emb)


def building_embedding_avg():
    emos = []
    cluster_avg_embeddings = []
    with open(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\financial_emotional_vocabulary.pkl','rb') as f:
        EMO_RESOURCES = pickle.load(f)

    for key in EMO_RESOURCES.keys():
        print(key)
        emos.append(key)
        print(EMO_RESOURCES[key])
        word_list = EMO_RESOURCES[key]
        avg_embedding = compute_avg_emb(word_list)
        print(avg_embedding)
        cluster_avg_embeddings.append(avg_embedding)


    df = pd.DataFrame()

    df['emotion']=emos
    df['avg_cluster_embeddin'] = cluster_avg_embeddings

    df.to_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\emotion_cluster_embeddings.csv")
