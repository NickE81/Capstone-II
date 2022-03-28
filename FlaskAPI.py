# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Function performing the whole word prediction process
def make_predictions(sentence):
    # Prepare the input for prediction
    x = text_prep(sentence)
    # Predict the next word using the predict function on our model
    preds = model.predict(x, verbose = 0)
    # Get the indices of the top n most likely next words
    top, scores = top_n(preds, 3)
    # Get the suggested words based on the indices
    suggs = get_words(top)
    return suggs, scores

# Preparing input text for next word prediction
def text_prep(text):
    # Split the input text into a list of words in the sentence
    text = text.lower().split()
    # Take the last (at most) 5 words for prediction, because the model can
    # not handle more than 5 words
    pred_text = text[-5:]
    # Create a list for the input in a form usable by the model
    x = np.zeros((1, MEMORY_LENGTH, number_of_words))
    # Populate the input array, where 1 in (i, j) represents the ith word
    # being the jth word in the unique word index
    for t, word in enumerate(pred_text):
        x[0, t, unique_word_index[word]] = 1
    # Return the input
    return x

# Get the indices of the top n most likely next words
def top_n(preds, n):
    # Make the preds array of type float
    preds = np.asarray(preds).astype('float64')
    # Take the log of preds
    preds_log = np.log(preds)
    # Then take e^preds_log
    preds_exp = np.exp(preds_log)
    # Find the percentage for each words
    preds = preds_exp / np.sum(preds_exp)
    # Find the indices of the top n most likely words
    words = (-preds).argsort()[0, 1:n+1]
    scores = -100 * np.sort(-preds)[0, 1:n+1]
    scores = [round(num, 2) for num in scores]
    # Return the list of the indices of the top n most likely words in
    # descending order
    return words, scores

def get_words(top):
    # Create a list for the suggested words
    suggs = []
    # For each top words index
    for i in top:
        # For each pair of word and index in the unique word index list
        for word, idx in unique_word_index.items():
            # If the unique word index matches the top words index
            if idx == i:
                # We have found the word we are looking for, add it to the
                # suggestions list
                suggs.append(word)
    # Return the suggestions list
    return suggs

model = load_model('tuned-model.h5')

# Get data
file = 'data.txt'
data = open(file).read().lower()

# Tokenize the data to a usable form
token = RegexpTokenizer(r'\w+')
words = token.tokenize(data)

# Get all the unique words, index them, and get the number of unique words
unique_words = np.unique(words)
unique_word_index = dict((word, i) for i, word in enumerate(unique_words))
# Note: # of words is 4289
number_of_words = len(unique_words)

# Want the model to predict the next word based on 5 previous words
MEMORY_LENGTH = 5

@app.route('/')
def home():
    global sentence, suggs, selected, user, table, records
    sentence = ""
    suggs = []
    selected = ""
    user = "Guest"
    table = pd.read_csv("records.csv")
    records = table[table.Name == user]
    records = [records.to_html(index = False)]
    return render_template('index.html', user = user, records = records)

@app.route('/predict', methods = ['POST'])
def predict():
    global sentence, suggs, records
    sentence = [x for x in request.form.values()][0]
    suggs, scores = make_predictions(sentence)
    records = table[table.Name == user]
    records = [records.to_html(index = False)]
    return render_template('prediction.html', sentence = sentence,
                           word1 = suggs[0], word2 = suggs[1],
                           word3 = suggs[2], score1 = scores[0], 
                           score2 = scores[1], score3 = scores[2],
                           user = user, records = records)

@app.route('/selection', methods = ['POST'])
def selection():
    global sentence, selected, suggs, records, table
    if request.form.get('choice1'):
        selected = request.form['choice1']
        option = "1"
    elif request.form.get('choice2'):
        selected = request.form['choice2']
        option = "2"
    elif request.form.get('choice3'):
        selected = request.form['choice3']
        option = "3"
    else:
        selected = request.form['other']
        option = "Other"
    table = table.append({'Name': user, 'Sentence': sentence, 'Word': selected,
                  'Position': option}, ignore_index = True)
    table.to_csv('records.csv', index = False)
    sentence = sentence + " " + selected
    suggs, scores = make_predictions(sentence)
    records = table[table.Name == user]
    records = [records.to_html(index = False)]
    return render_template('prediction.html', sentence = sentence,
                           word1 = suggs[0], word2 = suggs[1],
                           word3 = suggs[2], score1 = scores[0], 
                           score2 = scores[1], score3 = scores[2],
                           user = user, records = records)

@app.route('/change_user', methods = ["POST"])
def change_user():
    global sentence, suggs, selected, user, records
    sentence = ""
    suggs = []
    selected = ""
    user = request.form['new_user']
    records = table[table.Name == user]
    records = [records.to_html(index = False)]
    return render_template('index.html', user = user, records = records)
                           

@app.route('/reset', methods = ["POST"])
def reset():
    global sentence, suggs, selected, records
    sentence = ""
    suggs = []
    selected = ""
    records = table[table.Name == user]
    records = [records.to_html(index = False)]
    return render_template('index.html', user = user, records = records)


if __name__ == '__main__':
    app.run()