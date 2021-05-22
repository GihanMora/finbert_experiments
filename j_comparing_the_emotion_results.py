import ast

import pandas as pd

original_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\original_emo_profiles.csv",encoding='cp1252')
finbert_df = pd.read_csv(r"E:\Projects\Emotion_detection_gihan\pros_all_meta_reports_finbert.csv",encoding='utf-8')

pdf_list_original = []
for i,each_pdf in original_df.iterrows():
    pdf_list_original.append(each_pdf['pdf'])

pdf_list_finbert = []
for i,each_fpdf in finbert_df.iterrows():
    pdf_list_finbert.append(each_fpdf['pdf'])

dff = pd.DataFrame()

final_pdf_list = list(set(pdf_list_finbert).intersection(pdf_list_original))
for pdf in final_pdf_list:
    dict_row = {}
    dict_row['pdf'] = pdf
    print(pdf)
    original_profile = original_df.loc[original_df['pdf'] == pdf]
    op = ast.literal_eval(original_profile.presentation_all_e_p.values[0])
    op = {w+'_ori' : op[w] for w in op.keys()}
    print(op)
    dict_row.update(op)
    fin_profile = finbert_df.loc[finbert_df['pdf'] == pdf]
    fp = ast.literal_eval(fin_profile.presentation_all_e_p.values[0])
    fp = {w + '_fin': fp[w] for w in fp.keys()}
    print(fp)
    dict_row.update(fp)
    print(dict_row)
    dff= dff.append(dict_row,ignore_index=True)



dff.to_csv('emo_comparision.csv')




# print(original_df['pdf'])
# print(original_df['presentation_all_e_p'])
# print(finbert_df['presentation_all_e_p'])
