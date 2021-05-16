import pickle
import os

f_path = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\refined_dumps"

refined_dumps = os.listdir(f_path)
negatives = []
positives = []
for rd in refined_dumps:
    print(rd)
    if(rd in ['trust.pkl','surprise.pkl','joy.pkl','anticipation.pkl']):
        with open(os.path.join(f_path,rd),'rb') as f:
            EMO_RESOURCES = pickle.load(f)
            print(EMO_RESOURCES)
            positives.append(EMO_RESOURCES[rd[:-4]])
    else:
        with open(os.path.join(f_path, rd), 'rb') as f:
            EMO_RESOURCES = pickle.load(f)
            print(EMO_RESOURCES)
            negatives.append(EMO_RESOURCES[rd[:-4]])


common_dict = {}

print('*****refinement')
for rd in refined_dumps:
    print(rd)
    with open(os.path.join(f_path, rd), 'rb') as f:
        EMO_RESOURCES = pickle.load(f)
        common_dict[rd[:-4]] = EMO_RESOURCES[rd[:-4]]

print(common_dict)

# for e_key in common_dict:
#     if(e_key!='anger'):continue
#     if(e_key in ['anticipation','trust','joy','surprise']):
#         in_anger,in_sad,in_disgust,in_fear = [],[],[],[]
#         for e_word in common_dict[e_key]:
#             if(e_word in common_dict['anger']):
#                 in_anger.append(e_word)
#             if (e_word in common_dict['sad']):
#                 in_sad.append(e_word)
#             if (e_word in common_dict['disgust']):
#                 in_disgust.append(e_word)
#             if (e_word in common_dict['fear']):
#                 in_fear.append(e_word)
#         print('in_anger', in_anger)
#         print('in_sad', in_sad)
#         print('in_disgust', in_disgust)
#         print('in_fear', in_fear)
#     elif (e_key in ['anger', 'sad', 'disgust', 'fear']):
#         in_anticipation, in_trust, in_joy, in_surprise = [], [], [], []
#         for e_word in common_dict[e_key]:
#             if (e_word in common_dict['anticipation']):
#                 in_anticipation.append(e_word)
#             if (e_word in common_dict['trust']):
#                 in_trust.append(e_word)
#             if (e_word in common_dict['joy']):
#                 in_joy.append(e_word)
#             if (e_word in common_dict['surprise']):
#                 in_surprise.append(e_word)
#         print('in_anticipation',in_anticipation)
#         print('in_trust', in_trust)
#         print('in_joy', in_joy)
#         print('in_surprise', in_surprise)
#
to_remove_from_anger = ['caressing','obesity','bridgeport','enjoyable','showcases','entrepreneurs','rescues','delightful','yearning','befriended','showcasing','facilitating','securely','documenting','implementations','facilitates','sited']
to_remove_from_anticipation = ['demoted','outraged','groans','rebellious','documenting','paranoia','bellowed','entrepreneurs','prohibiting','insurrection','sarcastic','yanking','collapses','impatience','perpetrators','rejecting','dissatisfaction','catastrophic','ferocious','cynical','ousted','distraught','persecuted','middlesbrough','detrimental','menacing','thwarted','fearful','destroys','erroneously','dismay','condemning','tormented','grimace','infused','ominous','forbade']
to_remove_joy = ['spa','grasslands','southport','quivering','noteworthy','restricting','deities', 'alleviate', 'torino', 'zhejiang', 'florian', 'northampton','secluded', 'discouraged', 'tianjin','titanium', 'nils', 'oskar','promenade','fragmented','unease', 'crippled',  'dissatisfied','austrians','northamptonshire','dismay','discourage', 'relocating', 'tormented', 'exhibiting', 'horrific', 'anxious', 'moans', 'fearful', 'baffled','showcases' ,'forbade', 'groans', 'distraught','appalled','punishments', 'hesitant', 'despised', 'grimace',  'taunting', 'bewildered', 'curving', 'impatience', 'sarcastic', 'ambiguity', 'yanking', 'beheaded', 'intimidating', 'detrimental',  'ferocious', 'impoverished', 'persecuted','inspecting', 'showcasing', 'implementations', 'middlesbrough', 'ruining', 'destroys', 'infused', 'irritated', 'hideous', 'thwarted', 'bridgeport', 'decreases','pained','catastrophic', 'cynical', 'ousted', 'humiliated', 'caressing', 'outraged', 'insulting', 'dazzling', 'reiterated','insurrection', 'disgusted','radically', 'dissatisfaction', 'burnley', 'spearheaded' ]
to_remove_from_disgust = ['shimmering','progressing', 'pious','pulsing','decreed','astonishing','dazzling','yearning', 'sited', 'lauded', 'implementations', 'relocating'
,'northwards', 'caressing','delightful','markedly', 'middlesbrough','cadiz', 'showcasing', 'endeavors','glistening','facilitating', 'facilitates', 'securely','attaining',
'radically', 'remodeled','encourages', 'curving', 'enjoyable','imaginative', 'profoundly','inspecting', 'expansive', 'evaluating','extravagant', 'informally', 'redeveloped'
'southport','authoritarian', 'grasslands', 'ambiguity', 'sparkled', 'intimidating', 'outdated','protruding', 'disastrous', 'forbade', 'writhing', 'nils', 'cylindrical', 'reworked','exhibiting', 'tufts', 'pisa', 'freiburg', 'plump', 'sparkle', 'systematically', 'accolades','gleamed', 'sizable', 'migrating', 'delicately', 'decorate', 'northampton', 'creamy','northamptonshire','apical', 'ostensibly', 'supervise', 'warwickshire', 'encircled', 'filippo', 'landfill','oskar', 'declares']

to_remove_sad = ['attaining','yearning','recounted','encourages','sited','showcasing','implementations','acknowledges', 'northwards','specifies', 'redeveloped',
'reassured','enjoyable', 'securely', 'appointing', 'delightful', 'groans', 'lauded', 'caressing', 'astonishment','inspecting', 'relocating', 'cadiz','rescues', 'facilitating',
'asserting',  'facilitates', 'imaginative', 'spearheaded','shimmering', 'middlesbrough', 'befriended','fortifications','corpses', 'regaining','manipulating','appointing', 'plight',
'adolescence', 'tenderly', 'hilarious','magdalena','befriended']

to_remove_from_fear = [ 'shimmering','initiating', 'acknowledging', 'evaluating', 'endeavors', 'prohibiting','middlesbrough', 'forbade','asserting', 'befriended',
'northwards', 'glistening','remodeled', 'reassured', 'showcasing', 'championed', 'attaining','sited','accelerating','acknowledges','reiterated', 'courageous', 'imaginative',
 'disclose', 'implementations','relocating', 'showcases','encourages', 'empowered', 'delightful','lauded','redeveloped', 'markedly', 'recounted', 'facilitates', 'enjoyable']

to_remove_suprise = ['tormented','delicately', 'groans', 'lauded', 'showcases', 'thwarted', 'spearheaded','middlesbrough', 'caressing', 'antics',  'facilitates',
'coldly', 'destroys', 'baffled','showcasing', 'stupidity', 'northamptonshire','authoritarian', 'detrimental', 'irritating', 'punishments', 'beheaded', 'asserting','caressed',
 'exhibiting', 'curving', 'inspecting',
                          'sociologist', 'humorous', 'hilarious', 'kindness', 'trophies', 'austrians', 'revered',
                     'stimulated', 'attaining', 'staggering', 'sweetly', 'picturesque',
                     'chuckles', 'beautifully', 'radically', 'endeavors', 'northwards', 'decorate', 'acknowledges',
                     'sweetness', 'accomplishments', 'festivities', 'stiffly', 'celebrates', 'philanthropic',
                     'philanthropist', 'sincerity', 'showcased', 'admiration', 'enjoyable', 'pious'
                     ]
to_remove_from_trust = ['sarcastic','ousted', 'insulted', 'controversies', 'forbade', 'enjoyable', 'relocating', 'infused','hostility', 'modernized', 'coldly',
'quivering', 'outraged', 'undermine','pressured', 'beheaded', 'fearful', 'hampered', 'terrifying', 'yanking', 'harassed','ahmedabad','disturbances', 'prohibiting', 'contentious',
 'onslaught','discourage', 'disagreement', 'fraudulent', 'thwarted', 'destroys', 'dazzling', 'hesitant', 'dissatisfaction','cynical', 'tormented', 'persecuted', 'rejecting', 'sited',
  'pulsing','detrimental', 'shuddering', 'appalled', 'deterioration', 'radically','insurrection', 'northamptonshire', 'middlesbrough','intimidating', 'rochdale','disagreements', 'condemning', 'curving',
  'groans', 'influencing', 'insulting', 'disagree','impoverished', 'ferocious', 'ravaged', 'impatience','distrust', 'distraught', 'rouen', 'pained','bewildered'
'grasslands','priests', 'chapels','diplomats', 'mosques','evangelical', 'nils','perpetrators', 'discouraged','scriptures', 'religions', 'sermons','evangelical','preacher',
'unreliable','religions', 'parochial', 'restricting','seductive','dioceses',  'extravagant','preaching','astonished', 'austrians', 'cleansing', 'prayed','vocalists', 'churches',
'glistening','deities','crippled', 'mischievous', 'protagonists', 'theologian','informally','narratives', 'cocky', 'disagreed','championed', 'heroine', 'worshipped',
'northampton', 'spirituality','atheist', 'cadiz', 'catholicism','clergy', 'clergyman','villains',  'pious','resigning','sermon']

with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl','rb') as f:
    EMO_RESOURCES = pickle.load(f)
print(EMO_RESOURCES['trust'])

# for each_key in common_dict:
#     if(each_key!='trust'):continue
#     in_positives = []
#     in_negatives = []
#     print(each_key,[x for x in common_dict[each_key] if x not in to_remove_from_trust+EMO_RESOURCES['trust']])
#     # print(each_key, common_dict[each_key])
#
#     for each_w in common_dict[each_key]:
#         if(each_w in positives[0]):
#             in_positives.append(each_w)
#         if (each_w in negatives[0]):
#             in_negatives.append(each_w)
#     print('in positives',in_positives)
#     print('in negatives',in_negatives)

# in_pos ['cynical', 'emphasizing', 'appalled', 'asserting', 'caressing', 'paranoia', 'dazzling', 'grimace', 'championed', 'attaining', 'repairing', 'collapses', 'yanking', 'curving', 'perpetrators', 'enjoyable', 'bewildered', 'anxious', 'rebellious', 'outraged', 'dissatisfaction', 'ascribed', 'impatience', 'revolutionaries', 'stimulated', 'menacing', 'tormented', 'markedly', 'forbade', 'hurrying', 'showcases', 'entrepreneurs', 'thwarted', 'rescues', 'delightful', 'subtly', 'reiterated', 'provoke', 'redeveloped', 'rejecting', 'condemning', 'radically', 'provoked', 'hesitant', 'ominous', 'catastrophic', 'middlesbrough', 'lauded', 'detrimental', 'yearning', 'groans', 'befriended', 'showcasing', 'inspecting', 'distraught', 'insurrection', 'influencing', 'erroneously', 'facilitating', 'destroys', 'pulsing', 'ousted', 'dismay', 'taunting', 'prohibiting', 'relocating', 'modernized', 'securely', 'astonishing', 'ferocious', 'astonishment', 'spearheaded', 'infused', 'persecuted', 'documenting', 'implementations', 'facilitates', 'fearful', 'sited', 'sarcastic']

fixed_dict = {}
with open('E:\Projects\Emotion_detection_gihan\\from git\\nlp-emotion-analysis-core/src/models/emotions/emotions_plutchik.pkl','rb') as f:
    EMO_RESOURCES = pickle.load(f)

for each_key in common_dict:
    print(each_key)
    if (each_key == 'anger'): to_remove = to_remove_from_trust
    if (each_key == 'anticipation'): to_remove = to_remove_from_anticipation
    if (each_key == 'joy'): to_remove = to_remove_joy
    if (each_key == 'disgust'): to_remove = to_remove_from_disgust
    if (each_key == 'sad'): to_remove = to_remove_sad
    if (each_key == 'fear'): to_remove = to_remove_from_fear
    if (each_key == 'suprise'): to_remove = to_remove_suprise
    if (each_key == 'trust'): to_remove = to_remove_from_trust
    print(common_dict[each_key])
    print(len(common_dict[each_key]))
    fixed = [x for x in common_dict[each_key] if x not in to_remove]
    print(fixed)
    print(len(fixed))
    fixed_dict[each_key] = fixed
    fixeda = [x for x in common_dict[each_key] if x not in to_remove + EMO_RESOURCES[each_key]]
    print(fixeda)
    print(len(fixeda))


pkl_filename = r"E:\Projects\Emotion_detection_gihan\finbert_experiments\data_processed\high_quality_dumps\\financial_emotional_vocabulary.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(fixed_dict, file)
