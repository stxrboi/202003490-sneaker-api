FROM python:3.11.5
ENV DJANGO_SETTINGS_MODULE=sneakerApi.settings
ENV PYTHONBUFFERED 1
RUN mkdir /app
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY  ./app /app
EXPOSE 8000
CMD [ "python","manage.py","runserver","0.0.0.0:8000" ]
