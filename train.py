import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# opening the json file
with open('intents.json') as file:
    data = json.load(file)

# this stores the patterns from the json
training_sentences = []
# this stores the tag from json
training_labels = []
# this stores the labels from model
labels = []
# this is the response made by the model
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

print('Appending done!')
num_classes = len(labels)

print('\nDoing label encoding')
label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
training_labels = label_encoder.transform(training_labels)
print('\nDone label encoding')

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

print('\nDoing tokenization')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
print('\nDone with tokenization')

# Creating the network
print("\nBuilding the network")
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print('\nCompiling the network model')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('\nModel Summary')
model.summary()

epochs = 550
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

print("\nSaving the model!")

# Saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saving label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print('\nDone with saving model!')
print('\nLast line executed!')
