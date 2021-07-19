# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    # read data
    train = pd.read_csv("/Users/luisek/Documents/GitHub/elmos/datasets/train_2kmZucJ.csv")
    test = pd.read_csv("/Users/luisek/Documents/GitHub/elmos/datasets/test_oJQbWVk.csv")

    print(train.shape,test.shape)

    # distribution of postive vs negatives

    print(train["label"].value_counts(normalize=True))

    ## clean the text
    # remove some urls
    train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+','',x))
    test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

    # punctations
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{\}~'

    train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    # convert text to lowercase
    train['clean_tweet'] = train['clean_tweet'].str.lower()
    test['clean_tweet'] = test['clean_tweet'].str.lower()

    # remove numbers
    train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
    test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

    # remove whitespaces
    train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ' '.join(x.split()))
    test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

    # stemming? or finding the root of the word
    # using spacy language model

    nlp = spacy.load('en', disable=['parser','ner'])

    # function to lemmatize text
    def lemmatization(text):
        output = []
        for i in text:
            s = [token.lemma_ for token in nlp(i)]
            output.append(' '.join(s))
        return output

    train['clean_tweet'] = lemmatization(train['clean_tweet'])
    test['clean_tweet'] = lemmatization(test['clean_tweet'])

    print(train.head())
    # print(nlp(train['clean_tweet'][0]))
    # train['clean_tweet'] = lemmatization(train['clean_tweet'])

    # Preparing the elmo *cartoon* models

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
