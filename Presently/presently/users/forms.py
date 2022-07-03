from xml.parsers.expat import model
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django import forms
from .models import Video

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class Video_form(forms.ModelForm):
    class Meta:
        model = Video
        fields = ("caption","video")