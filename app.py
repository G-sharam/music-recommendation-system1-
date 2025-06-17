import pickle
import pandas as pd
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"


csv_path = r"d:\IIT Project\music predication system\streamlit\music.csv"
pkl_path = r"d:\IIT Project\music predication system\streamlit\df.pkl"


df = pd.read_csv(csv_path)

df.dropna(subset=['artist', 'song'], inplace=True)
df.reset_index(drop=True, inplace=True)

try:
    with open(pkl_path, "rb") as f:
        music = pickle.load(f)
except FileNotFoundError:
    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    music = df.copy()

music['combined_features'] = music['artist'] + ' ' + music['song']

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    if song not in music['song'].values:
        st.error("Sorry sir i can't found your song")
        return [], []

    cv = CountVectorizer()
    vectors = cv.fit_transform(music['combined_features'])

    index = music[music['song'] == song].index[0]
    song_vector = vectors[index]
    sim_scores = cosine_similarity(song_vector, vectors).flatten()

    similar_indices = sim_scores.argsort()[::-1][1:6]

    recommended_names = []
    recommended_posters = []

    for i in similar_indices:
        artist = music.iloc[i]['artist']
        title = music.iloc[i]['song']
        recommended_names.append(title)
        recommended_posters.append(get_song_album_cover_url(title, artist))
    
    return recommended_names, recommended_posters

st.header('Music Recommender System')
music_list = music['song'].values
selected_song = st.selectbox("Select your music", music_list)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_song)
    if names:
        cols = st.columns(5)
        for i in range(len(names)):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])
