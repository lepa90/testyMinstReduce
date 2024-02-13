from django.shortcuts import render
import tensorflow as tf
# Create your views here.
from nltk.tokenize import WhitespaceTokenizer 
from textblob import TextBlob
from django.http import JsonResponse
from django.views import View
import numpy as np
import noisereduce as nr
import librosa
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import soundfile as sf
import os
import mimetypes
from django.conf import settings
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import SpeechClient
import io
from io import BytesIO
from django.shortcuts import render
import json
from textblob import TextBlob
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


from django.http import JsonResponse
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

        # attempt authentication
from tweepy import OAuthHandler
import tweepy
from textblob import TextBlob
import re
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
from django.views import View

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from django.http import HttpResponse
from django.views import View

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from io import BytesIO
from nltk.tokenize import word_tokenize
class MNISTView(View):
    def get(self, request):
        # Load dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        max_examples = 10000
        train_images = train_images[:max_examples]
        train_labels = train_labels[:max_examples]

        # Create the model
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10)
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=5)

        # Make predictions
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(test_images)

        # Create a new figure and set the DPI
        fig = plt.figure(dpi=80)
        img = test_images[0]
        plt.title('Prediction: {}, Label: {}'.format(np.argmax(predictions[0]), test_labels[0]))
        plt.imshow(img, cmap=plt.cm.binary)

        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)

        # Create an HTTP response with the image
        response = HttpResponse(buf.getvalue(), content_type='image/png')
        return response


import tweepy

class TwitterClient(object):
    def __init__(self):
        consumer_key = 'O4ueOcR7xbmInLyGAlMGPTuCB'
        consumer_secret = 'x2YwgPHtLUd0nQXxzkhTp1uj14HniqloAa0YUYHzE2QRXiVDUh'
        access_token = '3993673420-fsdrGIc7LupxhslAScIEdNBVb2mYZDVZ9Nb0TEz'
        access_token_secret = '6Ci7vsotFEpvdZt4AmxICj0iZNm2hlsJyJT4iMfOF73Fg'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

import tweepy

import tweepy

class TwitterClient(object):
    def __init__(self):
        consumer_key = 'O4ueOcR7xbmInLyGAlMGPTuCB'
        consumer_secret = 'x2YwgPHtLUd0nQXxzkhTp1uj14HniqloAa0YUYHzE2QRXiVDUh'
        access_token = '3993673420-fsdrGIc7LupxhslAScIEdNBVb2mYZDVZ9Nb0TEz'
        access_token_secret = '6Ci7vsotFEpvdZt4AmxICj0iZNm2hlsJyJT4iMfOF73Fg'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, username, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.user_timeline(screen_name=username, count=count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepyException as e:
            # print error (if any)
            print("Error : " + str(e))

class TweetSentimentView(View):
    def get(self, request, *args, **kwargs):
        api = TwitterClient()
        print("Uruchomiono TwitterClient")  # Dodajemy instrukcję wydruku
        username = request.GET.get('query', 'realDonaldTrump')
        count = int(request.GET.get('count', 10))
        tweets = api.get_tweets(username=username, count=count)

        if len(tweets) > 0:
            ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
            positive_percentage = 100*len(ptweets)/len(tweets)

            ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
            negative_percentage = 100*len(ntweets)/len(tweets)

            neutral_percentage = 100*(len(tweets) - (len(ntweets) + len(ptweets)))/len(tweets)
        else:
            positive_percentage = 0
            negative_percentage = 0
            neutral_percentage = 0

        context = {
            'positive_tweets_percentage': positive_percentage,
            'negative_tweets_percentage': negative_percentage,
            'neutral_tweets_percentage': neutral_percentage,
        }
        return render(request, 'tweet-sentiment.html', context)


def index(request):
    return render(request, 'index.html')

def tweet(request):
    return render(request, 'tweet.html')

def audio(request):
    return render(request, 'audio.html')
def minst(request):
    return render(request, 'minst.html')
@csrf_exempt
def upload_video(request):
    try:
        if request.method == 'POST':
            video_file = request.FILES['data']

            # Zapisz plik wideo na serwerze
            with open('video.mp4', 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            # Wczytaj plik wideo
            video = VideoFileClip("video.mp4")
        
            # Wyodrębnij ścieżkę audio
            audio = video.audio
            
            # Zapisz ścieżkę audio do pliku WAV
            audio.write_audiofile("audio.wav")

            # Wczytaj plik audio
            audio = AudioSegment.from_file("audio.wav")

            # Eksportuj do formatu mp3
            audio.export("audio.mp3", format="mp3")

            return HttpResponse("Konwersja zakończona pomyślnie!")
        else:
            return HttpResponse("Błąd: metoda żądania musi być POST.")
    except Exception as e:
        return HttpResponse(f"Wystąpił błąd: {str(e)}")




@csrf_exempt
def transcribe_audio_stream(request):
    client = SpeechClient()

    audio = request.FILES['audio_data'].read()
    audio = speech.RecognitionAudio(content=audio)
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.MP3,
        language_code="en-US",
        audio_channel_count=2,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    responses = client.streaming_recognize(streaming_config, audio)

    transcript = ''
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        print("Transcript: {}".format(result.alternatives[0].transcript))
        transcript = result.alternatives[0].transcript

    return JsonResponse({'transcript': transcript})
@csrf_exempt
def tokenize_transcript(transcript):
    tokens = word_tokenize(transcript)
    print("Tokens: ", tokens)
    return tokens
@csrf_exempt
def transcribe_audio(request):

    if 'audio_data' not in request.FILES:
        return JsonResponse({'error': 'No audio file received'})

    client = speech.SpeechClient()

    audio = request.FILES['audio_data'].read()
    audio = speech.RecognitionAudio(content=audio)

    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz = 44100,
        language_code="en-US",
        audio_channel_count=2,
    )

    response = client.recognize(config=config, audio=audio)

    transcript = ''
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        transcript = result.alternatives[0].transcript

    # Tokenizacja transkryptu za pomocą nowej metody
    tokens = tokenize_transcript(transcript)

    # Zapisz transkrypt i tokeny do sesji
    request.session['transcript'] = transcript
    request.session['tokens'] = tokens

    # Zwróć JSON z transkryptem i tokenami
    context = {'transcript': transcript, 'tokens': tokens}
    return render(request, 'transcribe_audio.html', context)

@csrf_exempt
def reduce_noise(request):
    if request.method == 'POST':
        audio_file = request.FILES['data']
        # Odczytaj plik audio jako dane binarne
        file_content = audio_file.read()
        # Zapisz dane do pliku tymczasowego
        with open('temp.wav', 'wb') as f:
            f.write(file_content)
        # Odczytaj plik audio za pomocą librosa
        noisy_data, sr = librosa.load('temp.wav')
        # Zredukuj szum
        reduced_noise = nr.reduce_noise(y=noisy_data, sr=sr)

        sf.write('reduced_noise.wav', reduced_noise, sr)
        
        return JsonResponse({'status': 'success', 'message': 'Plik audio został pomyślnie przesłany i zredukowano szum.'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Nieobsługiwana metoda żądania.'})
def download_file(request):
    # Zdefiniuj pełną ścieżkę do pliku
    file_path = os.path.join(settings.MEDIA_ROOT, 'reduced_noise.wav')
    # Zdefiniuj mimetypes dla pliku
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or 'application/octet-stream'
    # Ustaw nagłówki dla pliku
    response = HttpResponse(content_type=mime_type)
    # Ustaw nagłówek Content-Disposition
    response['Content-Disposition'] = "attachment; filename=%s" % os.path.basename(file_path)
    # Odczytaj plik
    with open(file_path, 'rb') as f:
        response.write(f.read())
    # Zwróć odpowiedź
    return response