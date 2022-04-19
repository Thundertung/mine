import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


app = Flask(__name__)
picFolder = os.path.join('static','pics')

app.config['UPLOAD_FOLDER'] = picFolder
f = open('group_name.json')
group_name = json.load(f)
f = open('group_id.json')
group_id = json.load(f)
f = open('group_encoded_dict.json')
group_encoded_dict = json.load(f)

f = open('job_dict.json')
job_dict = json.load(f)
app = Flask(__name__)
model = tf.keras.models.load_model('career_recommendation_last.h5')

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def info2groupdf(jobList):
    groupLst = []
    for job in jobList:
        if ' 'in job:
            groupLst.append(group_id[group_name[job.replace(' ','_').lower()]])
        else:
            groupLst.append(group_id[group_name[job.lower()]])
    return groupLst



def generate_career_seq(model, seed_text, n_words):

  text = []
  
  for _ in range(n_words):
    encoded = info2groupdf(seed_text)
    encoded = pad_sequences([encoded], maxlen = 10, truncating='post',padding='post')
    predict_x=model.predict(encoded) 
    y_predic=np.argmax(predict_x,axis=1)

    predicted_word = ''
    for word, index in group_encoded_dict.items():
      if index == y_predic:
        predicted_word = word
        break 
    seed_text.append(predicted_word)
    text.append(predicted_word)

  return seed_text 



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    return render_template("recommendation.html")


@app.route('/info')
def recjob():
    return render_template('insertInfo.html')

@app.route('/career')
def career():
    return render_template('career.html')


@app.route('/career2')
def career2():
    return render_template('career2.html')

@app.route('/predict',methods=['POST'])
def predict():


    str_features = [str(x) for x in request.form.values()]

    prediction = generate_career_seq(model,str_features ,1)
    return render_template('recommendation.html', prediction_text='Your career path should be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, port=80)