f = open('E:\Projects\Emotion_detection_gihan\\finbert_experiments\\financial phrasebank\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt','r')

lines = f.readlines()
sentences = []
sentiments = []
for each_line in lines:
    try:
        row = each_line.strip().split(' .@')
        print(row)
        if(len(row)==2):
            sentences.append(row[0])
            sentiments.append(row[1])
    except Exception:
        continue

import pandas as pd

df = pd.DataFrame()
df['sentence'] = sentences
df['sentiment'] = sentiments
print(df['sentiment'].unique())
sentiment_ids = []
for i,rr in df.iterrows():
    if(rr['sentiment']=='neutral'):
        sentiment_ids.append(0)
    elif (rr['sentiment'] == 'positive'):
        sentiment_ids.append(1)
    elif (rr['sentiment'] == 'negative'):
        sentiment_ids.append(2)

df['sentiment_id']=sentiment_ids


df.to_csv("E:\Projects\Emotion_detection_gihan\\finbert_experiments\\financial phrasebank\\processed_fpbank.csv")


