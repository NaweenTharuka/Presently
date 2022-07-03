from django.core.exceptions import ValidationError

def file_size(value):
    filesize=value.size
    if filesize>209715200:
        raise ValidationError("maximum size is 10 mb")