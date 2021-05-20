# import these modules
import pickle

from nltk.stem import PorterStemmer

porter = PorterStemmer()

print(porter.stem("insecurities"))
print(porter.stem("development"))
print(porter.stem("trouble"))

stemmed_dict = {}
with open(r'E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\financial_emotional_vocabulary.pkl','rb') as f:
    EMO_RESOURCES = pickle.load(f)

for key_e in EMO_RESOURCES:
    print(key_e,len(EMO_RESOURCES[key_e]))
    tokens_list = EMO_RESOURCES[key_e]
    tokens_list_stemmed = []
    for each_t in tokens_list:
        tokens_list_stemmed.append(porter.stem(each_t))

    print(EMO_RESOURCES[key_e])
    print(tokens_list_stemmed)
    print(len(list(set(tokens_list_stemmed))))
    stemmed_dict[key_e] = list(set(tokens_list_stemmed))


pkl_filename = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\\financial_emotional_vocabulary_stemmed.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(stemmed_dict, file)