FROM ubuntu:18.04
WORKDIR /app

COPY . /app

RUN sudo apt update
RUN sudo apt upgrade -y

RUN apt install python3
RUN apt install python3-pip

RUN pip install -r users/requirement.txt

CMD ["python", "./users/manage.py","runserver"]
EXPOSE 8000