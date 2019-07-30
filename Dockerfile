FROM ubuntu:18.04
RUN apt-get update &&\
    apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools &&\
    apt install -y python3-venv
#ADD . /app
#WORKDIR /app
#RUN pip install wheel
#RUN pip install -r requirements/dev.txt
#EXPOSE 8000
#CMD ["gunicorn", "-b", "0.0.0.0:8000", "wsgi:app"]
