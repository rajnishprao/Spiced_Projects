'''
convert song lyrics to Bag of Word vectors using TfidfVectorizer
'''

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizer(x):
    all_lyrics = []
    artist_name = []
    path = './songs/'
    for artist in x:
        for file in os.listdir(path):
            if artist in file and '.txt' in file:
                with open(path + file, 'r') as f:
                    all_lyrics.append(f.read())
                    artist_name.append(artist)

    artist_name = pd.factorize(artist_name)[0]
    vect = TfidfVectorizer()
    vected_lyrics = vect.fit_transform(all_lyrics)
    print('All lyrics vectorized')
    return vected_lyrics, artist_name, vect
