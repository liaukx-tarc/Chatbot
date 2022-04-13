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

words=[]
classes = []
testData = []
ignore_words = []
data_file = open('menu_intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        testData.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(testData), "test data")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)


pickle.dump(words,open('menu_words.pkl','wb'))
pickle.dump(classes,open('menu_classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
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


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0, nesterov=False, name="SGD")
sgd = optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1, validation_data=(train_x,train_y))
model.save('menu_chatbot_model.h5', hist)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss', 'val loss'])
plt.show()

print("model created")
