import time

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
import json
import random

#menu
menu_model = load_model("Menu/menu_chatbot_model.h5")
menu_intents = json.loads(open('Menu/menu_intents.json').read())
menu_words = pickle.load(open('Menu/menu_words.pkl', 'rb'))
menu_classes = pickle.load(open('Menu/menu_classes.pkl', 'rb'))

# order
order_model = load_model("Food Order/order_chatbot_model.h5")
order_intents = json.loads(open('Food Order/order_intents.json').read())
order_words = pickle.load(open('Food Order/order_words.pkl', 'rb'))
order_classes = pickle.load(open('Food Order/order_classes.pkl', 'rb'))

# customer service
cust_model = load_model("Customer Service/customer_service_chatbot_model.h5")
cust_intents = json.loads(open('Customer Service/customer_service_intents.json').read())
cust_words = pickle.load(open('Customer Service/customer_service_words.pkl', 'rb'))
cust_classes = pickle.load(open('Customer Service/customer_service_classes.pkl', 'rb'))

# example
example_model = load_model('Inti/example_chatbot_model.h5')
example_intents = json.loads(open('Inti/example_intents.json').read())
example_words = pickle.load(open('Inti/example_words.pkl', 'rb'))
example_classes = pickle.load(open('Inti/example_classes.pkl', 'rb'))

# inti
model = example_model
intents = example_intents
words = example_words
classes = example_classes
isChooseFunc = False
isConfirm = False
previousResp = None
foodOrdering = None
currentModel = 0
startTime = None
endTime = None
foodCart = []
isQuit = False


# preprocess the input
def clean_up_sentence(sentence):
    # separate all words in the input
    sentence_words = nltk.word_tokenize(sentence)

    # change all the words in words array to original word and lowercase
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# create the input data
def bow(sentence, words, show_details=True):
    # preprocess the user input
    sentence_words = clean_up_sentence(sentence)

    # create the input feature
    input = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # if the word have inside the words data change the input to 1
                input[i] = 1

                # print the input found
                if show_details:
                    print("found in input: %s" % w)
    return (np.array(input))


# predict the class with the model
def predict_class(sentence, model):
    # create the input data
    p = bow(sentence, words, show_details=False)
    # start to predict
    res = model.predict(np.array([p]))[0]
    # set the error probabilitys
    error_probability = 0.5
    # if the predict probability too low ignore it
    results = [[i, r] for i, r in enumerate(res) if r > error_probability]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    if len(results) != 0:
        for r in results:
            # add it into the result
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    else:
        # if not thing in the result add the noanswer class
        return_list.append({"intent": "noanswer"})

    return return_list


# get the response of chatbot
def getResponse(predict_result, intents_json):
    tag = predict_result[0]['intent']
    list_of_intents = intents_json['intents']
    global isChooseFunc
    global isConfirm
    global previousResp
    global foodOrdering
    global foodCart
    global model
    global intents
    global words
    global classes
    global currentModel
    global isQuit

    if isConfirm:
        if tag == "confirm" or tag == "cancel":
            if tag == "confirm":
                foodCart.append(foodOrdering)

            for i in list_of_intents:
                # compare the tag with the json intent file
                if (i['tag'] == tag):
                    # random choose a responses
                    result = random.choice(i['responses'])
                    isConfirm = False
                    break
        else:
            result = "Bot: Please answer my question first. Thanks\n\n" + "[" + time.ctime(time.time()) + "] Bot: " + previousResp + "\n\n"

    elif isChooseFunc:
        if tag == "1" or tag == "2" or tag == "3":
            for i in list_of_intents:
                # compare the tag with the json intent file
                if (i['tag'] == tag):
                    # random choose a responses
                    result = random.choice(i['responses'])

                    if tag == '1':
                        model = menu_model
                        intents = menu_intents
                        words = menu_words
                        classes = menu_classes
                        currentModel = 1

                    elif tag == '2':
                        model = order_model
                        intents = order_intents
                        words = order_words
                        classes = order_classes
                        currentModel = 2

                    else:
                        model = cust_model
                        intents = cust_intents
                        words = cust_words
                        classes = cust_classes
                        currentModel = 3

                    isChooseFunc = False
                    break
        else:
            result = "Please select an option to proceed.\n\n" + "[" + time.ctime(time.time()) + "] Bot: " + previousResp

    else:
        for i in list_of_intents:
            # compare the tag with the json intent file
            if (i['tag'] == tag):
                # random choose a responses
                result = random.choice(i['responses'])

        if tag == "goodbye":
            isQuit = True

        if tag == "menu" or tag == "order" or tag == "customer service":
            if tag == 'menu':
                model = menu_model
                intents = menu_intents
                words = menu_words
                classes = menu_classes
                currentModel = 1

            elif tag == 'order':
                model = order_model
                intents = order_intents
                words = order_words
                classes = order_classes
                currentModel = 2

            elif tag == 'customer service':
                model = cust_model
                intents = cust_intents
                words = cust_words
                classes = cust_classes
                currentModel = 3

        if currentModel == 2 or currentModel == 3:
            if tag == "food cart":
                i = 0
                result += '\n'
                for f in foodCart:
                    result += '\n\t\t\t\t' + str(i + 1) + ". " + f
                    if i != len(foodCart) - 1:
                        result += '\n'
                    i += 1

            if currentModel == 2:
                if tag == "chicken rice" or tag == "fried rice" or tag == "nasi lemak" or tag == "fish porridge" or tag == "laksa" or tag == "fried noodle" or tag == "noodle soup" or tag == "teh tarik" or tag == "coffee" or tag == "milo":
                    isConfirm = True
                    foodOrdering = tag

                elif tag == "confirm" or tag == "cancel":
                    result = "Not thing to confirm."

    previousResp = result
    return result

def chatbot_response(text):
    # predict the result
    predict_result = predict_class(text, model)
    # show what have inside the result
    print("".join(str(p) for p in predict_result))

    # get the response
    res = getResponse(predict_result, intents)
    return res


# Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    global  startTime
    global  endTime

    startTime = time.time()
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "[" + time.ctime(time.time()) + "] You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = chatbot_response(msg)
        endTime = time.time()
        ChatLog.insert(END, "[" + time.ctime(time.time()) + "] Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        print((endTime - startTime) * 1000, "ms\n")
        if isQuit:
            time.sleep(3)
            quit()


base = Tk()
base.title("Hello")
base.geometry("800x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=776, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=770)
EntryBox.place(x=128, y=401, height=90, width=665)
SendButton.place(x=6, y=401, height=90)

def main():
    global isChooseFunc
    global previousResp
    isChooseFunc = True
    ChatLog.config(state=NORMAL)
    ChatLog.config(foreground="#442265", font=("Verdana", 12))
    startQues = "[" + time.ctime(time.time()) + "] Bot: Please Choose the Function you want\n\n" + "\t\t\t\t 1. Food Menu\n\n" + "\t\t\t\t 2. Food Order\n\n" + "\t\t\t\t 3. Customer Service\n\n"
    ChatLog.insert(END, startQues)
    previousResp = startQues
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)



if __name__ == "__main__":
    main()

base.mainloop()
