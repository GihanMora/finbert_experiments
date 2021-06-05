import pandas as pd

path = r"E:\Projects\A_Idiom_detection_gihan\idiom_detection_nlp\building_emotional_embeddings\EmotionLines\procesd_emotionlines_emo_ids.csv"

import sys
import pdfplumber
import os
import pandas as pd
from os.path import dirname as up
# one_up = up(__file__)
# sys.path.insert(0, one_up)

sys.path.insert(1, 'E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core')
# sys.path.insert(1, 'E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core\\src\\')

import pickle
import src.core.emotions.emotion_extractor as emotion_extractor
import src.utils.text_processor as text_utils
import src.core.summary.keyphrase_extractor as keyphrase_extractor
import src.core.clinical_info.clinical_info_extractor as clinical_info_extractor


def load_emotion_dictionaries():
    # with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl', 'rb') as f:
    #     EMOTION_MAP = pickle.load(f)
    with open(
            r'E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\financial_emotional_vocabulary_stemmed.pkl',
            'rb') as f:
        EMOTION_MAP = pickle.load(f)
    with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/intensifier_vocab_v2.pkl', 'rb') as f:
        INTENSIFIER_MAP = pickle.load(f)
    with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/negation_vocab_v2.pkl', 'rb') as f:
        NEGATION_MAP = pickle.load(f)
    with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/clinical_info/physical.pkl', 'rb') as f:
        PHYSICAL = pickle.load(f)

    EMO_RESOURCES = {'EMOTIONS': EMOTION_MAP,
                     'NEGATION': NEGATION_MAP,
                     'INTENSIFIERS': INTENSIFIER_MAP,
                     'PHYSICAL': PHYSICAL}

    return EMO_RESOURCES



results_df = pd.DataFrame()
df = pd.read_csv(path)
print(df.columns)
print(df['emotion'].unique())
EMO_RESOURCES = load_emotion_dictionaries()
for i,row in df.iterrows():
    # if(i>20):continue
    row_dict = row.to_dict()
    # print()
    sentence = row['utterance']
    clean_text_1 = text_utils.clean_text(sentence)
    emotion_profile, emo_seq = emotion_extractor.get_emotion_profile_per_post(clean_text_1, EMO_RESOURCES)
    print(emotion_profile)


    row_dict.update(emotion_profile)
    print(row_dict)
    results_df = results_df.append(row_dict, ignore_index=True)


results_df.to_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\evaluations\twitter_emotion_evaluations.csv")

