import numpy as np
from keras.models import load_model
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Preparing input text for next word prediction
def text_prep(text):
    # Split the input text into a list of words in the sentence
    text = text.lower().split()
    # Take the last (at most) 5 words for prediction, because the model can not handle more than 5 words
    pred_text = text[-5:]
    # Create a list for the input in a form usable by the model
    x = np.zeros((1, MEMORY_LENGTH, number_of_words))
    # Populate the input array, where 1 in (i, j) represents the ith word being the jth word in the unique word index
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
    preds = (-preds).argsort()[0, 1:n+1]
    # Return the list of the indices of the top n most likely words in descending order
    return preds

def get_words(top):
    # Create a list for the suggested words
    suggs = []
    # For each top words index
    for i in top:
        # For each pair of word and index in the unique word index list
        for word, idx in unique_word_index.items():
            # If the unique word index matches the top words index
            if idx == i:
                # We have found the word we are looking for, add it to the suggestions list
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
    global sentence, suggs, selected
    sentence = ""
    suggs = []
    selected = ""
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    global sentence, suggs
    sentence = [x for x in request.form.values()][0]
    # Prepare the input for prediction
    x = text_prep(sentence)
    # Predict the next word using the predict function on our model
    preds = model.predict(x, verbose = 0)
    # Get the indices of the top n most likely next words
    top = top_n(preds, 3)
    # Get the suggested words based on the indices
    suggs = get_words(top)
    return render_template('prediction.html', sentence = sentence, word1 = suggs[0], word2 = suggs[1], word3 = suggs[2])

@app.route('/selection', methods = ['POST'])
def selection():
    global sentence, selected
    if request.form['other'] == "":
        selected = request.form['submit_button']
        return render_template('selected.html', sentence = sentence, word1 = suggs[0], word2 = suggs[1], word3 = suggs[2],
                            selected = "The final sentence is: " + sentence + " " + selected)
    else:
        selected = request.form['other']
        return render_template('selected.html', sentence = sentence, word1 = suggs[0], word2 = suggs[1], word3 = suggs[2],
                            selected = "The final sentence is: " + sentence + " " + selected)

@app.route('/reset', methods = ["POST"])
def reset():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()