'''
Main file to interact with user
'''

import os
import re
import sys
import pandas as pd
import numpy

from lyrics_scraper import get_lyrics
from song_vectorizer import vectorizer
from silly_bayes import silly_bayes

def welcome_message():
    print('''
Welcome to the Guess The Artist game!
Enter the names of as many artists as you like
and train a Naive Bayes model to recognise their lyrics.
    ''')

def user_inputs():
    '''
    accepts artist names from user inputs
    '''
    user_artists = []
    i = True
    while i:
        artist = input("Please enter an artist's name or type done when finished: ")
        if artist.strip().lower() == 'done':
            i = False
        else:
            user_artists.append(artist)
            print(f'Selected artist: {artist}')
    return user_artists

def guess_artist(guess, vect, model):
    clean_guess = re.sub('[\n\-\?\.\,\(\)]', ' ', guess)
    clean_guess = re.sub('[\']', '', clean_guess)
    clean_guess = [clean_guess] # whats this doing?
    vect_guess = vect.transform(clean_guess)
    prediction = model.predict_proba(vect_guess)
    print('This is perhaps a song from:')
    return prediction

if __name__ == '__main__':
    # welcome message
    welcome_message()
    # get user input and scrape web for lyrics
    user_artists = user_inputs()
    scrapped_lyrics = get_lyrics(user_artists)
    # vectorize scrapped lyrics and train model
    vected_lyrics, artist_name, vect = vectorizer(scrapped_lyrics)
    model = silly_bayes(vected_lyrics, artist_name)
    # guess the artist
    guess = input(f'Enter a sample of lyrics from one of your chosen artists:')
    prediction = guess_artist(guess, vect, model)
    df = pd.DataFrame(prediction.round(3), columns=user_artists)
    print('The certainity of the guess is:', df)
