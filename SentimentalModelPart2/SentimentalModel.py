### Part 2 of learning sentimental models
### First one used a naive bayes algorithm and just used the probabilities of each 
### word independently being positive or negative
### This one aims to improve accuracy by using word embeddings (continuous bag of words)
### 

import os
import numpy as np
import pandas as panda
import json
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Dense, Flatten
from sklearn.linear_model import LogisticRegression
import TextProcessing as txt

class SentimentalModel():
    WindowSize = 3

    def __init__(self):
        self.valid = False
        self.Features = {}
        self.trainingData = []

        #input and training and output directories
        training = "training"
        output = "output"
        self.sample = 2000

        if not os.path.exists(output):
            os.mkdir(output)
        
        if not os.path.exists(training):
            os.mkdir(training)

        if os.path.exists("Parameters.txt"):
            with open("Parameters.txt", "r") as jsonFile:
                file = json.load(jsonFile)
                self.sample = int(file["trainingSample"])

       #load training data
        trainingModels = os.listdir(training)
        if len(trainingModels) != 0:
            if self.loadTraining(trainingModels, self.sample):
                print("loaded training data")
                self.TrainModel()
                self.valid = True

    def loadTraining(self, files, sample):
        columnnames = ['polarity', 'id', 'post_datetime', 'query', 'user', 'tweet']
        print("Loading Training Data")
        try:
            for file in files:
                with open("training/" + file, "r") as csvfile:
                    dataSets = panda.read_csv(csvfile, names = columnnames)

                    ## randomly sample 2000 rows for positive and negative
                    positive = dataSets[dataSets["polarity"] == 4]
                    negative = dataSets[dataSets["polarity"] == 0]

                    positive = positive["tweet"].head(sample).tolist()
                    negative = negative["tweet"].head(sample).tolist()

                    self.trainingData = [positive, negative]

            print("Finished Data Loading")
            return True
        except:
            return False

    ## Prepare the data for the model
    def PrepData(self):
        modelX = []
        modelY = []

        positiveData = self.trainingData[0]
        negativeData = self.trainingData[1]

        for statement in positiveData[0:2000]:
            statement = txt.PreprocessText(statement)

            modelX.append(statement)
            modelY.append(4)
        for statement in negativeData[0:2000]:
            statement = txt.PreprocessText(statement)

            modelX.append(statement)
            modelY.append(0)
        
        return modelX, modelY
    
    def TrainModel(self):
        print("Preparing Data")
        xData, yData = self.PrepData()
        print("Data Prep finished")
        print("Training Embedding Model")
        self.CBOWModel(xData, 2)
        print("Finish Training Embedding Model")
        print("Training Classifier")
        self.ClassifyModel(xData, yData)
        print("Finished Training Model")

    def PredictSentiment(self, sentence):
        ## Convert sentence to vector
        sequence = self.SentenceToVector(sentence)

        return "Positive" if self.LR.predict([sequence])[0] == 4 else "Negative"


    def ClassifyModel(self, xTrain, yTrain):
        self.LR = LogisticRegression()
        
        xTrain = [self.SentenceToVector(x) for x in xTrain]
        self.LR.fit(xTrain, yTrain)

    def SentenceToVector(self, sentence):
        ## Preprocess sentence
        sentence = txt.PreprocessText(sentence).split(" ")

        ## Get word embeddings
        embeddings = self.model.layers[1].get_weights()[0]
        
        X = np.zeros(100) ## initialize vector

        ## Loop over words
        count = 0
        for word in sentence:
            if word in self.tokenizer.word_index:
                X += embeddings[self.tokenizer.word_index[word]]
                count += 1
        
        if count != 0:
            X /= count
        
        return X
    
    def CBOWModel(self, data, windowSize):
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(data)
        wordIndicies = self.tokenizer.index_word

        X, Y = [], []

        ## list of sentence sequences 
        sequences = self.tokenizer.texts_to_sequences(data)

        ## Loop over each sequence in the sequences
        for sequence in sequences:
            # For each sequence, get the context "words" around each target "word"
            for i in range(windowSize, len(sequence) - windowSize):
                context = sequence[i - windowSize:i] + sequence[i:i+windowSize]
                target = sequence[i]
                X.append(context)
                Y.append(target)

        X = np.array(X)
        Y = np.array(Y)

        ## size of the vocabulary
        vocabSize = len(wordIndicies) + 1

        ## size of each word embedding
        embeddingSize = 100

        ## Building the neural network
        ## input layer is expecting 4 "words"
        input_layer = Input(shape=(2 * windowSize,))
        embeddingLayer = Embedding(input_dim=vocabSize, output_dim = embeddingSize)(input_layer)
        flattenLayer = Flatten()(embeddingLayer)
        outputLayer = Dense(units=vocabSize, activation='softmax')(flattenLayer)

        model = Model(inputs=input_layer, outputs = outputLayer)
        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")

        model.fit(X, Y, epochs=10)
        self.model = model


                








