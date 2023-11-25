FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py ./
COPY handle_recs.py ./
COPY worker.py ./
COPY data_processing ./data_processing
COPY frontend ./frontend
COPY static ./static
COPY templates ./templates
ENV REDISCLOUD_URL=redis://redis:6379
ENV DATABASE_URL=mongodb://mongodb:27017/

