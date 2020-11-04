'''
Script scrapes song lyrics and saves it into a folder
'''

import os
import time
import re
import requests
from bs4 import BeautifulSoup as bs


def get_lyrics(x):
    '''from a list of artist names, lyrics scraped from one page of songs
    '''
    if not os.path.exists('songs'):
        os.makedirs('songs')

    base_url = 'http://www.metrolyrics.com/'
    artist_names = []
    for each in x:
        # artist names
        artist = each.lower().strip().split(' ')
        artist_name = '_'.join(artist)
        artist_names.append(artist_name)
        # artist path
        artist_path = '-'.join(artist) + '-lyrics.html'
        path = base_url + artist_path
        # getting links
        all_songs = requests.get(path)
        all_songs_parsed = bs(all_songs.text, 'html.parser')
        results = all_songs_parsed.find_all(attrs={'class':'songs-table compact'})[0]
        songs = results.find_all('a')
        artist_links = []
        for i in range(len(songs)):
            artist_links.append(songs[i].get('href'))
        # getting song names
        regex = r'https:\/\/www\.metrolyrics\.com\/(\S+)'
        song_names = []
        for each in artist_links:
            song = re.findall(regex, each)[0]
            song = song.split('-')
            artist_size = len(artist) + 1
            song = song[:-artist_size]
            song = '_'.join(song)
            song = artist_name + '_' + song
            song_names.append(song)
        # saving lyrics
        for i in range(len(songs)):
            song = requests.get(artist_links[i])
            song_parsed = bs(song.text, 'html.parser')
            song_lyrics = song_parsed.find_all(attrs={'class':'js-lyric-text'})[0]
            song_hell = song_lyrics.find_all('p')
            lyrics = ''
            for each in song_hell:
                lyrics += each.text
            final_lyrics = re.sub(r'[\n\-\?\.\,\(\)]', ' ', lyrics)
            final_lyrics = re.sub(r'[\']', '', final_lyrics)
            file = './songs/' + song_names[i] + '.txt'
            with open(file, 'w') as f:
                f.write(lyrics)
        print('Song lyrics have been scrapped off the www')
        return artist_names
