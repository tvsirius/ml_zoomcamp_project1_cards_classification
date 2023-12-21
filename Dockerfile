FROM python:3.10.12-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["cardclass.py", "./"]
COPY ["model/cardmodel.tflite", "./model/"]
COPY ["templates/main.html", "./templates/"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "cardclass:app"]
