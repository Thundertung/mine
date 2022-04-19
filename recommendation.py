import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json



def info2groupdf(jobList):
  f = open('group_name.json')
  group_name = json.load(f)
  f = open('group_id.json')
  group_id = json.load(f)
  f = open('group_encoded_dict.json')
  group_encoded_dict = json.load(f)
  groupLst = []
  for job in jobList:
    groupLst.append(group_id[group_name[job.replace(' ','_')]])
  return groupLst



def generate_career_seq(model, seed_text, n_words):
  
  f = open('group_name.json')
  group_name = json.load(f)
  f = open('group_id.json')
  group_id = json.load(f)
  f = open('group_encoded_dict.json')
  group_encoded_dict = json.load(f)
  text = []
  
  for _ in range(n_words):
    #seed_text_space = seed_text.split(' ')
    #print(seed_text_space)
    #encoded = map_job_dict(seed_text)
    encoded = info2groupdf(seed_text)
    #encoded = tokenizer.texts_to_sequences([seed_text])[0]
    #encoded = group_encoded_map(encoded)
    encoded = pad_sequences([encoded], maxlen = 10, truncating='post',padding='post')
    #print(encoded)

    #y_predic = model.predict_classes(encoded)
    predict_x=model.predict(encoded) 
    #print(encoded)
    y_predic=np.argmax(predict_x,axis=1)

    predicted_word = ''
    for word, index in group_encoded_dict.items():
      if index == y_predic:
        predicted_word = word
        break 
    seed_text.append(predicted_word)
    text.append(predicted_word)
  #return ' '.join(text)
  return seed_text 



# fix code to combine both degree and job

