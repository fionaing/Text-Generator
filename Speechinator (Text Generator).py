#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:14:01 2022

@author: ianing
"""
import numpy as np
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#File Formatting ------------------------------

file = open("Crime and Punishment.txt").read() #/Users/ianing/Desktop/Text Generator/

def formats(file):
    '''
    Formats file by making all the words lowercase and tokenizing them (splitting them up into individual words).
    Ex: "Hi, im Bob." --> ["Hi", ",", "i'm", "Bob", "."]
    (similar to .split(), but seperates punctuation from words)
    If the words is a "stop word"(common/unhelpful word), it is removed. 
    '''
    file = file.lower() #makes text lowercase
    
    tokens = RegexpTokenizer(r'\w+').tokenize(file) #splits into individual words
    
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)
    

file = formats(file)

chars = sorted(list(set(file))) #finds all unique characters

char_num = {} #dictionary to turn characters into numbers
for n, l in enumerate(chars): #counts number of times unique character appears
    char_num[l] = n #l is letter, n is number

num_char = {} #dictionary to turn the numbers back into characters
for n, l in enumerate(chars):
    num_char[n] = l #l is letter, n is number
#Sequencing and Converting------------------------------
'''
seq_length is the length of the sequence of characters we want to look at before predicting a character
Goes through entire list of inputs to convert characters into numbers. 
Each sequence starts with the next letter in the text.
Ex: seq_length = 4, text = "I'm Fiona"

  X (seq)   y (predict)
[I, ', m, ][F]
[', m, , F][i]
[m, , F, i][o]
etc
'''
def sequencing(seq_length, input_len):
    X0 = []
    y0 = []
    
    for i in range(0, input_len - seq_length, 1): #going through all inputs, converts to numbers
        in_seq = file[i:i + seq_length] #input is current characters to sequence length
        out_seq = file[i + seq_length] #output is inital character plus totsl sequence length
        
        #adds converted characters to our lists
        y0.append(char_num[out_seq])
        X0.append([char_num[char] for char in in_seq])
    return X0, y0
    
X0, y0 = sequencing(100, len(file))
n_patterns = len(X0)

#Turn Into Arrays ------------------------------
X = np.reshape(X0, (n_patterns, 100, 1)) #reshapes array, 100 is the seq_length
X = X/float(len(chars))
y = np_utils.to_categorical(y0) #one-hot encode y

#gets a random number to get character from XO to act as a seed (starts it off) for the model
rand_n = np.random.randint(0, len(X0) - 1)
start_words = X0[rand_n]


#LSTM Model Creation ------------------------------
'''
Sequential allows you to create a layer at a time, though its limited by one input and one output.
We have 3 layers.
'''
model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) #256 is number of neurons
model.add(Dropout(.2)) #prevents overfitting
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128))
model.add(Dropout(.2))
model.add(Dense(y.shape[1], activation='softmax')) #dense layer that outputs probability

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam') #compile consfigures model so we can start training

#Checkpoints ------------------------------
'''
Saves weights in increments to make things easier. 
'''
f_path = "weights.hdf5"
checkpoint = ModelCheckpoint(f_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
call_backs = [checkpoint]


#LSTM Model Fit and Train ------------------------------
model.fit(X, y, epochs=1, batch_size=256, callbacks=call_backs) #increase epochs for better result, but takes longer

weights = "weights.hdf5"
model.load_weights(weights)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

def generate_seed(X0, num_char):
    #gets a random number to get character from XO to act as a seed (starts it off) for the model
    rand_n = np.random.randint(0, len(X0) - 1)
    start = X0[rand_n]
    
    #for printing out the random seed, it turns rand_words into words from numbers:
    print("Seed:")
    print("\"", ''.join([num_char[val] for val in start]), "\"")
    return start

start_words = generate_seed(X0, num_char)

#Generate Text ------------------------------
'''
Converts seed into float values. 
It will go through chosen range of characters and then predict what characters come next based off of the seed. 
Converts output to characters which it then puts back into pattern, which are the seed + previous characters.  
'''
for i in range(100):
    x = np.reshape(start_words, (1, len(start_words), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = num_char[index]

    sys.stdout.write(result)

    start_words.append(index)
    start_words = start_words[1:len(start_words)]
    
print("Generated text:")
print("\"", ''.join([num_char[value] for value in start_words]), "\"") #resulting seed + predicted text

