import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import random
import pickle

import colorama

colorama.init()
from colorama import Fore, Style, Back

with open('intents.json') as file:
    data = json.load(file)


def chat_model():
    # Load the trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoder object
    with open('label_encoder.pickle', 'rb') as encoder:
        label_encoder = pickle.load(encoder)

    # Parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLACK_EX + "User: " + Style.RESET_ALL, end="")
        input_t = input()
        if input_t.lower() == 'quit':
            break

        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(training_sentences)
        # word_index = tokenizer.word_index
        # sequences = tokenizer.texts_to_sequences(training_sentences)

        result = model.predict(
            keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([input_t]),
                truncating='post', maxlen=max_len)
        )

        tag = label_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot: " +
                      Style.RESET_ALL, np.random.choice(i['responses'])
                      )


print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" +
      Style.RESET_ALL
      )
chat_model()
