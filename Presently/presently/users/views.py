from array import array
from http.client import HTTPResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from .forms import UserRegisterForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import Video_form
from .models import Video
from django.template import Template, Context
import datetime
from . import grammerchecker
from . import emotiondetectionvideo
# from . import emotiondetectionaudio
import myprosody as mysp
import pickle
from . import myprosody

def test(request):
    p="publicspeech" 
    c="F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\users\\myprosody" 
    var2= myprosody.myspgend(p,c)
    var1= myprosody.myprosody(p,c)
    context= {
        'var1': var1,
        'var2': var2
    }
    return render(request, 'users/test.html',context)

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
    stt = grammerchecker.speechText
    result = grammerchecker.result
    context= {
        'result': result,
        'stt': stt
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
    all_video=Video.objects.all()

    return render(request, 'users/emotionaudio.html',{"all":all_video}) 

def handgestures(request):
    all_video=Video.objects.all()

    return render(request, 'users/handgestures.html',{"all":all_video})     

def emotionvideo(request):
    emotion= emotiondetectionvideo.emo
    arr= emotiondetectionvideo.arr1
    context= {
        'maxemotion': emotion,
        'predictions': arr
    }
    return render(request, 'users/emotionvideo.html',context) 

def emotionaudioprosody(request):
    p="publicspeech" 
    c="F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\users\\myprosody" 
    myspsyl= myprosody.myspsyl(p,c)
    mysppaus= myprosody.mysppaus(p,c)
    myspsr= myprosody.myspsr(p,c)
    myspatc= myprosody.myspatc(p,c)
    myspst= myprosody.myspst(p,c)
    myspod= myprosody.myspod(p,c)
    myspbala= myprosody.myspbala(p,c)
    myspf0mean= myprosody.myspf0mean(p,c)
    myspf0sd= myprosody.myspf0sd(p,c)
    myspf0med= myprosody.myspf0med(p,c)
    myspf0min= myprosody.myspf0min(p,c)
    myspf0max= myprosody.myspf0max(p,c)
    myspf0q25= myprosody.myspf0q25(p,c)
    myspf0q75= myprosody.myspf0q75(p,c)
    myspgend= myprosody.myspgend(p,c)
    mysppron= myprosody.mysppron(p,c)
    prosody= myprosody.myprosody(p,c)
    
    context= {
        'myspsyl': myspsyl,
        'mysppaus': mysppaus,
        'myspsr': myspsr,
        'myspatc': myspatc,
        'myspst': myspst,
        'myspod': myspod,
        'myspbala': myspbala,
        'myspf0mean': myspf0mean,
        'myspf0sd': myspf0sd,
        'myspf0med': myspf0med,
        'myspf0min': myspf0min,
        'myspf0max': myspf0max,
        'myspf0q25': myspf0q25,
        'myspf0q75': myspf0q75,
        'myspgend': myspgend,
        'mysppron': mysppron,
        'prosody': prosody
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


@login_required()
def profile(request):
    return render(request, 'users/profile.html') 

