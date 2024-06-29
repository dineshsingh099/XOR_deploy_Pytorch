FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY src /app/src

WORKDIR /app

RUN chmod -R 777 /app/src
RUN chmod +x /app/src/run_model.sh

RUN pip install -r /app/src/requirements.txt 

ENV PYTHONPATH=${PYTHONPATH}:/app/src

CMD ["sh", "/app/src/run_model.sh"]