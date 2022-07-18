from array import array
from http.client import HTTPResponse
import os
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from .forms import UserRegisterForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import Video_form
from .models import Video
from django.template import Template, Context
import datetime
import pickle
from . import pipeline

def home(request):
    all_video=Video.objects.all()
    if request.method == "POST":
        form = Video_form(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, f'Your video was successfully uploaded')
            return redirect('feedback')
    else:
        form=Video_form()        
    return render(request, 'users/home.html',{"form":form,"all":all_video})

def contactus(request):
    return render(request, 'users/contactus.html')

def aboutus(request):
    return render(request, 'users/aboutus.html')

def pronuciation(request):
    pronuciation = pipeline.transcription
    print(pipeline.transcription)
    grammersuggest = pipeline.grammer_corrected
    context= {
        'pronuciation': pronuciation,
        'grammersuggest': grammersuggest
    }
    return render(request, 'users/pronuciation.html',context)  

def warningemotionsaudio(request):
    return render(request, 'users/unusualemotionswarning.html')  

def slidecheckerdashboard(request):
    return render(request, 'users/slidecheckerdashboard.html') 

def colorcubechecker(request):
    return render(request, 'users/colourcube.html')  

def grammercheckerslides(request):
    return render(request, 'users/grammerchecker.html')          

def feedback(request):
    all_video=Video.objects.all()

    return render(request, 'users/feedback.html',{"all":all_video}) 

def emotionaudio(request):
    emotionaudiosum = pipeline.emotion_logits_audio
    context= {
        'emotionaudiosum': emotionaudiosum
    }
    return render(request, 'users/emotionaudio.html',context) 

def handgestures(request):
    all_video=Video.objects.all()

    return render(request, 'users/handgestures.html',{"all":all_video})     

def emotionvideo(request):
    emotionvideosum = pipeline.emotion_logits_video
    context= {
        'emotionvideosum': emotionvideosum,
    }
    return render(request, 'users/emotionvideo.html',context) 

def emotionaudioprosody(request):
    prosodysum = pipeline.prosody_context
    context= {
        'prosodysum': prosodysum,
    }
    return render(request, 'users/prosody.html',context)    

def overallfeedback(request):
    return render(request, 'users/overallfeedback.html')                 

def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Hi {username}, your account was created successfully')
            return redirect('home')
    else:
        form = UserRegisterForm()

    return render(request, 'users/register.html', {'form': form})

def test(request):
    path="users/pose_detected/"  
    video_list =os.walk(path)   
    return render(request,'users/handgestures.html', {'videos': video_list}) 
 

@login_required()
def profile(request):
    return render(request, 'users/profile.html') 

