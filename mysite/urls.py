"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from apka import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('nor/', views.reduce_noise, name='nor'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('download_file/', views.download_file, name='download_file'),
    path('transcribe_audio/', views.transcribe_audio, name='transcribe_audio'),
    path('transcribe_audio_stream/', views.transcribe_audio_stream, name='speech'),
    path('minst/', views.minst, name="minst" ),
    path('audio/', views.audio, name="audio" ),
    path('tweet-sentiment/', views.TweetSentimentView.as_view(), name='tweet-sentiment'),
    path('mnist/', views.MNISTView.as_view(), name='mnist'),    
]
