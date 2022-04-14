import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------

# preprocess the data
# store all the words
words = []

# store all the classes
classes = []

# store the testing data [words, classes]
testData = []

# store the words want to ignore
ignore_words = []

# open the intents file
data_file = open('inti_intents.json').read()
intents = json.loads(data_file)

# for loop each pattern sentences in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # separate each word from the sentence
        w = nltk.word_tokenize(pattern)
        # add into the words array
        words.extend(w)
        # create test data with the sentence and class
        testData.append((w, intent['tag']))

        # add the new class to classes array
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# change all the words in words array to original word and lowercase
# also remove the duplicate word and ignore words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# sorted the words and classes array
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# print out all the data
# combination between patterns sentence and class
print(len(testData), "test data")
# all the classes
print(len(classes), "classes", classes)
# all words, vocabulary
print(len(words), "unique lemmatized words", words)

# save the words and classes data to pkl file
pickle.dump(words, open('inti_words.pkl', 'wb'))
pickle.dump(classes, open('inti_classes.pkl', 'wb'))

# -----------------------------------------------------------------------------------------------------

# create training data
# array of [Input(the array show the sentence have what word), Output(the array show the array maybe what class)]
training = []
# the empty array of all output class
output_empty = [0] * len(classes)

# each data in test data
for data in testData:

    # the empty array to save the sentence have what word
    input = []

    # the array to save all the word in the sentence
    words_in_sentence = data[0]

    # change all the words in words array to original word and lowercase
    words_in_sentence = [lemmatizer.lemmatize(word.lower()) for word in words_in_sentence]

    # go through all the word in the words array (all the words appear in pattern sentence)
    # if the word have inside the current sentence change add 1 to the array to represent it have this feature
    # if not have this word add 0 to the array
    for w in words:
        input.append(1) if w in words_in_sentence else input.append(0)

    # create the output data, change the current tag to 1, other tag remain 0
    output_row = list(output_empty)
    output_row[classes.index(data[1])] = 1

    # add the data to training data
    training.append([input, output_row])

# random shuffle the training data and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - class
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# -----------------------------------------------------------------------------------------------------------------
# create Neural Network model

# using the sequential model
model = Sequential()

# create input layer with all the feature(all word in pattern sentence)
# create first hidden layer with 128 Neuron and using relu activation
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# Dropout use is to randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
model.add(Dropout(0.5))

# create second hidden layer with 64 Neuron and using relu activation
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# create output layer for all the class with the softmax activation to predict
model.add(Dense(len(train_y[0]), activation='softmax'))

# show the model with print
model.summary()

# Compile model and set learning rate to 0.05
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False, name="SGD")
sgd = optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# give the data to model and start training, repeat 250 epochs
hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1, validation_data=(train_x,train_y))

# save the model
model.save('inti_chatbot_model.h5', hist)

# plot the graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Loss & Accuracy')
plt.ylabel('Loss & Accuracy')
plt.xlabel('Epochs')
plt.legend(['loss', 'val loss', 'accuracy', 'val_accuracy'])
plt.show()

print("model created")