import pandas as pd
import os

files_path = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\term_emb_label\\"

files_list = os.listdir(files_path)

df_list = []

for f in files_list:
    full_path = os.path.join(files_path,f)
    print(full_path)
    df = pd.read_csv(full_path)
    df_list.append(df)
print(files_list)

all_dfs = pd.concat(df_list, axis=0, ignore_index=True)

all_dfs.to_csv(r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\term_emb_label\all_emo.csv")
