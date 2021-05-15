import pandas as pd
import ast
from datetime import datetime
df = pd.read_csv(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\finbert_raw_embeddings.csv')
# print(df.head())
df = df.dropna()
import numpy as np
# from scipy import spatial
#
# dataSetI = [1, 0, -1]
# dataSetII = [1,-1, 0]
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
# print(result)
from sklearn.metrics.pairwise import cosine_similarity
dis = cosine_similarity([[1, 0, -1]], [[1,-1, 0]])
# print(dis)


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

def get_nearest_neighbours(word):
    t1 = datetime.now()
    tuples = []
    for i,row_e in df.iterrows():
        if(row_e['token']==word):
            print('found')
            term_vec = row_e['embeddings']
            for j,row_d in df.iterrows():
                # print(row_d)
                if(('[PAD]' in row_d['token']) or ('#' in row_d['token']) or ('unused' in row_d['token']) or (len(row_d['token'])==1) or (row_d['token'].isnumeric())or (check_cardinalities(row_d['token']))): continue
                dis = cosine_similarity([ast.literal_eval(term_vec)], [ast.literal_eval(row_d['embeddings'])])
                # print([row_e['tokens'],row_d['tokens'],dis])
                tuples.append([row_e['token'],row_d['token'],dis])
    print('ok')
    s_tup = sorted(tuples, key=lambda x: x[2])
    neaarest_neighbs = []
    for i,m in enumerate(s_tup[::-1]):
        # print(m)
        if(i<100):
            neaarest_neighbs.append(m[1])

    t2 = datetime.now()
    diff = t2-t1
    print('time',diff)
    print(neaarest_neighbs)

def check_cardinalities(term):
    if((term[-2:] in ['rd','th','nd','st']) and (term[:-2].isnumeric())):
        return True
    if ((term[-1:] in ['s']) and (term[:-1].isnumeric())):
        return True
    else:
        return False




def get_nearest_neighbours_for_word_list(word_list):
    t1 = datetime.now()
    tuples = []

    term_vec = compute_avg_emb(word_list)
    for j,row_d in df.iterrows():
        # print(row_d)
        if(('[PAD]' in row_d['token']) or ('#' in row_d['token']) or ('unused' in row_d['token']) or (len(row_d['token'])==1) or (row_d['token'].isnumeric()) or (check_cardinalities(row_d['token']))): continue
        dis = cosine_similarity([term_vec], [ast.literal_eval(row_d['embeddings'])])
        # print([row_e['tokens'],row_d['tokens'],dis])
        tuples.append([word_list,row_d['token'],dis])
    print('ok')
    s_tup = sorted(tuples, key=lambda x: x[2])
    neaarest_neighbs = []
    for i,m in enumerate(s_tup[::-1]):
        # print(m)
        if(i<50):
            neaarest_neighbs.append(m[1])

    t2 = datetime.now()
    diff = t2-t1
    print('time',diff)
    print(neaarest_neighbs)
    return neaarest_neighbs

# get_nearest_neighbours_for_word_list(['trust','faith'])
# get_nearest_neighbours('trust')
# get_nearest_neighbours('distrust')
# print(check_cardinalities('2222rd'))
# print(check_cardinalities('5th'))