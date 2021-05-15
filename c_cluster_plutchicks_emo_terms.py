import pandas as pd
import ast
from b_get_nearest_neighbours_finbert import get_nearest_neighbours_for_word_list
from datetime import datetime
df = pd.read_csv(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\finbert_raw_embeddings\finbert_raw_embeddings.csv')
# print(df.head())
df = df.dropna()
import numpy as np

from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
import numpy as np
import pickle
import sys
with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl','rb') as f:
    EMO_RESOURCES = pickle.load(f)
# EMOTION_MAP = EMO_RESOURCES['EMOTIONS']
emos = ['anticipation','disgust','joy','sad','fear','trust','surprise']
for emo in emos[6:7]:
    anger_terms = EMO_RESOURCES[emo.strip()]
    print(emo)
    print('plutchik token count :',len(anger_terms))

    embeddings = []
    for word in anger_terms:
        row = df.loc[df['token'] == word]
        if(len(row)>0):
            embeddings.append([word,ast.literal_eval(row['embeddings'].tolist()[0])])
            # embeddings.append([word,ast.literal_eval(row['embeddings'][0])])
        # for i,row_e in df.iterrows():
        #     if(row_e['token']==word):
        #         print(word)
        #         embeddings.append([word,row_e['embeddings']])

    # print(len(embeddings))
    # for em in embeddings:
    #     print(em)
    print('total tokens :',len(embeddings))
    X = np.array([x[1] for x in embeddings])
    clf = KMeansConstrained(
        n_clusters=5,
        size_min=3,
        size_max=9,random_state=42)



    kmeans = clf.fit(X)
    # kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    clusters = kmeans.labels_.tolist()

    dff = pd.DataFrame()

    dff['word'] = [w[0] for w in embeddings]
    dff['embedding'] = [e[1] for e in embeddings]
    dff['cluster'] = clusters

    clusters_dict = {key:[] for key in dff['cluster'].unique()}
    for i,rr in dff.iterrows():
        clusters_dict[rr['cluster']].append(rr['word'])



    for each_c in clusters_dict:
        print(each_c,clusters_dict[each_c])
        out_neigh = get_nearest_neighbours_for_word_list(clusters_dict[each_c])
        f = open(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\\"+str(emo)+"\\"+str(each_c)+".txt",'w+')
        for each_n in out_neigh:
            f.write(each_n+'\n')





# original kmeans
# 0 ['abandoned', 'abandonment', 'neglected', 'rejection']
# 2 ['abuse', 'aggressive', 'angry', 'awful', 'beating', 'bitch', 'bloody', 'bomb', 'brutal', 'crushed', 'crushing', 'damn', 'deadly', 'fatal', 'fight', 'fighting', 'hit', 'horrible', 'jerk', 'killing', 'nasty', 'prick', 'prison', 'prisoner', 'punch', 'punishment', 'scar', 'scream', 'screaming', 'shit', 'shooting', 'shot', 'slash', 'smash', 'stab', 'sting', 'terrible', 'torture', 'vicious', 'victim', 'violent', 'whip', 'worse', 'worst', 'wound', 'judgement', 'pressure', 'grip', 'bitter']
# 3 ['accusation', 'aggression', 'agitated', 'agitation', 'annoyance', 'annoyed', 'annoying', 'bitterness', 'bothering', 'bully', 'complain', 'complained', 'complaint', 'confinement', 'cursing', 'defiance', 'deny', 'deprivation', 'discrimination', 'distracting', 'explode', 'furiously', 'harmful', 'holocaust', 'hostile', 'inappropriate', 'insulting', 'irritating', 'noisy', 'obstacle', 'offended', 'pissed', 'powerless', 'pretending', 'raging', 'restriction', 'sarcasm', 'suspicious', 'tease', 'teasing', 'unacceptable', 'violently', 'wrongly', 'frustration', 'impatience', 'irritated', 'enraged', 'provoked', 'boiling']
# 5 ['accused', 'anger', 'bad', 'criminal', 'curse', 'darkness', 'disease', 'enemy', 'evil', 'lying', 'poison', 'rob', 'witch', 'cross']
# 7 ['adverse', 'conflict', 'delay', 'deterioration', 'difficulty', 'disaster', 'dismay', 'dispute', 'disruption', 'distress', 'distressed', 'disturbance', 'disturbed', 'dominate', 'domination', 'feud', 'fraud', 'hostage', 'intruder', 'involvement', 'rebellion', 'recession', 'revolution', 'saber', 'shaky', 'shortage', 'sore', 'strained', 'struggle', 'tension', 'tremor', 'turmoil', 'uncertain', 'unhappy', 'violation', 'wreck', 'strain', 'upset']
# 6 ['arrogant', 'betray', 'betrayal', 'burden', 'cursed', 'destroyed', 'destroying', 'destructive', 'distracted', 'hardened', 'homeless', 'injustice', 'invasion', 'jealous', 'mortality', 'poverty', 'ruined', 'ruthless', 'sabotage', 'screwed', 'selfish', 'sneak', 'spite', 'steal', 'stigma', 'stolen', 'sucked', 'sucks', 'tyrant', 'unfair', 'wasted', 'rude', 'paranoid']
# 4 ['assault', 'attack', 'attacking', 'offense', 'offensive']
# 1 ['blame', 'disagree', 'dislike', 'disliked', 'distrust', 'hate', 'hating', 'misunderstanding', 'reject', 'tolerate', 'unpleasant']
# 8 ['confront', 'insult', 'threat', 'threaten', 'threatening']
# 9 ['hysterical', 'insane', 'lunatic', 'mad', 'madness', 'possessed', 'ridiculous', 'silly', 'stupid']
