FROM python:3.8.5

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/api

ARG PIP_EXTRA_INDEX_URL

ADD ./api /opt/api/
RUN pip install --upgrade pip
RUN pip install -r /opt/api/requirements.txt
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt omw-1.4 wordnet stopwords

RUN chmod +x /opt/api/run.sh
RUN chown -R ml-api-user:ml-api-user ./


USER ml-api-user


EXPOSE 8001

CMD ["bash", "./run.sh"]
