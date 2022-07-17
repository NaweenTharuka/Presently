from wsgiref.validate import validator
from django.db import models

# from presently.users.validator import file_size
from .validator import file_size

# Create your models here.
class Video(models.Model):
    caption = models.CharField(max_length=100)
    video=models.FileField(upload_to="videos",validators=[file_size])
    def __str__(self) -> str:
        return self.caption