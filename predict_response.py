#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


# words to be igonred/omitted while framing the dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    # tokenize the user_input
    user_input_words = nltk.word_tokenize(user_input)    
    # convert the user input into its root words : stemming
    user_input_words = [stemmer.stem(word.lower()) for word in user_input_words]
    # Remove duplicacy and sort the user_input
    user_input_words = sorted(list(set(user_input_words)))
    # Input data encoding : Create BOW for user_input
    for word in words:
        bag.append(1) if word in user_input_words else bag.append(0)
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extract the class from the predicted_class_label
   predicted_class = classes[predicted_class_label]

   # now we have the predicted tag, select a random response

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # choose a random bot response
        bot_response = random.choice(intent['responses'])
    
        return bot_response
    

print("Hi I am Stella, How Can I help you?")

while True:

    # take input from the user
    user_input = input('Type you message here : ')
    response = bot_response(user_input)

    response = bot_response(user_input)
    print("Bot Response: ", response)