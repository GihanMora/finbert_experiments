import os
import pickle

TEENTHS = ['tenth','eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth','nineteenth']
TENTHS = ['twentieth','thirtieth', 'fortieth', 'fiftieth', 'sixtieth','seventieth', 'eightieth', 'ninetieth']
HUNDREDTH = ['hundredth','hundred'] # HUNDREDTH not s
ONES = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine']
TENS = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty','ninety']

numeric_strings = TEENTHS+TEENTHS+HUNDREDTH+ONES+TENS
emos = ['anger','anticipation','disgust','joy','sad','fear','trust','surprise']
emotion = emos[7]
f_path = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\\"+emotion
text_files = os.listdir(f_path)
# print(text_files)
with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl','rb') as f:
    EMO_RESOURCES = pickle.load(f)
# EMOTION_MAP = EMO_RESOURCES['EMOTIONS']

plutchick_vocab = EMO_RESOURCES[emotion]
print('plutchick ',len(plutchick_vocab))

vocabulary = []

for txt_f in text_files:
    f = open(os.path.join(f_path,txt_f),"r")
    word_list = [w.strip() for w in f.readlines()]
    vocabulary.extend(word_list)

print(vocabulary)
print(len(vocabulary))
vocabulary = list(set(vocabulary))
print('set',vocabulary)
print(len(vocabulary))

vocabulary = [term for term in vocabulary if term not in numeric_strings]
print('set',vocabulary)
print(len(vocabulary))

print("terms in plutchick but not in ours")
left_vocab = list(set(plutchick_vocab)-set(vocabulary))
print(left_vocab)
print(len(left_vocab))

refined = {emotion:vocabulary+left_vocab}

pkl_filename = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\refined_dumps\\"+emotion+".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(refined, file)

