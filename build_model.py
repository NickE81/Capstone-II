import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
import pickle
from keras_tuner import RandomSearch

# Defining functions:
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
    preds = (-preds).argsort()[0, :n]
    # Return the list of the indices of the top n most likely words in descending order
    return preds

# Find the words that correspond to the indices above
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

# Func called by keras_tuner to find optimal model
def build_model(hp):
    # Build a Sequential model
    model = Sequential()
    # Add a Long-Short Term Memory (LSTM) layer
    model.add(LSTM(hp.Int('LSTM nodes', min_value = 32, max_value = 128, step = 16),
        activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (MEMORY_LENGTH, number_of_words)))
    # Add a dense layer with softmax activation representing the output layer
    model.add(Dense(number_of_words, activation = 'softmax'))
    # Add a learning rate selection
    lr = hp.Float('Learning rate', min_value = 0.005, max_value = 0.02, step = 0.005)
    # Create a stochastic gradient descent optimizer
    opt = SGD(learning_rate = lr)
    # Compile the model with the SGD optimizer, categorical crossentropy loss, and accuracy as it's metric
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    

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
# Create lists to store sets of previous words, and their corresponding next word
prev_words = []
next_words = []
# Populate these lists with all the data retrieved from the txt file
for j in range(len(words) - MEMORY_LENGTH):
    prev_words.append(words[j:j + MEMORY_LENGTH])
    next_words.append(words[j + MEMORY_LENGTH])


# Create lists that will store the inputs and labels used by the model
# The inputs are each of shape 5 (MEMORY_LENGTH) x number_of_words, with a 1 in (i, j) representing 
# ith word in the previous words string being the jth word in the unique word index (dict of words)
X = np.zeros((len(prev_words), MEMORY_LENGTH, number_of_words))
# The labels are each a list of size number_of_words, with a 1 in the ith position meaning that the next word 
# is the ith word in the unique word index (dict of words), and a 0 everywhere else
Y = np.zeros((len(next_words), number_of_words))
# Populate the inputs array, and the labels array
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[each_word]] = 1
# Note: The dataset consists of 44967 total records

# Choose whether you'd like to train a new model, or test the accuracy of the current one
print('Would you like to: \n1. Train the model with the given hyperparameters\n2. Check accuracy of the model with given hyperparameters')
print('3. Search for the best model on a range of possible hyperparameters\n4. Predict the next word given an input')
choice = input()

# 1. If training new model
if choice == '1':
    # Build a Sequential model
    model = Sequential()
    # Add a Long-Short Term Memory (LSTM) layer with n nodes
    model.add(LSTM(112, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (MEMORY_LENGTH, number_of_words)))
    # Add a dense layer with softmax activation representing the output layer
    model.add(Dense(number_of_words, activation = 'softmax'))
    # Create a stochastic gradient descent optimizer
    opt = SGD(learning_rate = 0.02)
    # Print the model summary
    model.summary()
    # Compile the model with the SGD optimizer, categorical crossentropy loss, and accuracy as it's metric
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Fit the model on the data with 10% validation split, run for 15 epochs with a batch size of 32, and save its history
    hist = model.fit(X, Y, validation_split = 0.1, epochs = 40, batch_size = 32, verbose = 1).history
    # Save the model in an h5 file
    model.save('tuned-model.h5')
    # Save the history in a pickle file
    pickle.dump(hist, open('tuned-model.pkl', 'wb'))

# 2. If checking the accuracy of the current model
elif choice == '2':
    # Load the history pickle file
    hist = pickle.load(open("tuned-model.pkl", "rb"))
    # Plot the training and validation accuracy by epoch, and label and legend the graph
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('Accuracy of next word prediction model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.show()

# 3. If building a model using keras_tuner to find optimal parameters
elif choice == '3':
    # Build the tuner that will look for the best model
    tuner = RandomSearch(build_model, objective = 'val_accuracy', max_trials = 5, directory = 'next_word_output', project_name = 'next_word_prediction')
    # Perform search for the best model
    tuner.search(X, Y, validation_split = 0.1, epochs = 40, batch_size = 32)
    # Retrieve the best trained model
    models = tuner.get_best_models(num_models = 1)
    model = models[0]
    # Save the model in a h5 file
    model.save('tuned-model.h5')

# 4. If predicting next word from inputs
elif choice == '4':
    # Load the model
    model = load_model('tuned-model.h5')
    # This variable determines whether to continue or not
    stop = False
    # While we want to continue
    while (not stop):
        # Prompt the user for an input text, and save it
        print('Please enter the input text, and press enter')
        text = input()
        # Prepare the text for prediction
        x = text_prep(text)
        # Predict the next word using the predict function on our model
        preds = model.predict(x, verbose = 0)
        # Get the indices of the top n most likely next words
        top = top_n(preds, 3)
        # Get the suggested words based on the indices
        suggs = get_words(top)
        # Print the user's input
        print('Your input: ' + text)
        # Print the suggested words
        print('Suggested words: ')
        print(suggs)
        
        # Prompt the user to see if they want to continue
        print('Would you like to go again? Y or N?')
        choice = input()
        # If the user wants to continue
        if choice == 'Y':
            # Do nothing, i.e. leave stop as False
            pass
        # If the user does not want to continue
        elif choice == 'N':
            # Set the stop to True, terminate the program
            stop = True
        