FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip install wheel
RUN pip install -r requirements/dev.txt
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "wsgi:app"]
