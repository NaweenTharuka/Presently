from django.urls import path
from . import views
from django.contrib.auth import views as auth_view

urlpatterns = [
    path('', views.home, name='home'),
    path('contactus/', views.contactus, name='contactus'),
    path('aboutus/', views.aboutus, name='aboutus'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('feedback/', views.feedback, name='feedback'),
    path('feedback/slide-checker', views.slidecheckerdashboard, name='slidecheckerdashboard'),
    path('feedback/slide-checker/color-cube', views.colorcubechecker, name='colorcube'),
    path('feedback/slide-checker/grammer-blur-checker', views.grammercheckerslides, name='grammer'),
    path('feedback/summery', views.overallfeedback, name='overallfeedback'),
    path('feedback/pronuciation-vocubalry-errors', views.pronuciation, name='pronuciation'),
    path('feedback/emotion/audio', views.emotionaudio, name='emotionaudio'),
    path('feedback/emotion/audio/warnings', views.warningemotionsaudio, name='warningemotionaudio'),
    path('feedback/emotion/video', views.emotionvideo, name='emotionvideo'),
    path('feedback/emotion/video/hand-gestures', views.handgestures, name='handgestures'),
    path('feedback/emotion/audio/prosody', views.emotionaudioprosody, name='emotionaudioprosody'),
    path('login/', auth_view.LoginView.as_view(template_name='users/login.html'), name="login"),
    path('logout/', auth_view.LogoutView.as_view(template_name='users/logout.html'), name="logout"),
    path('test/', views.test, name='test'),

]
