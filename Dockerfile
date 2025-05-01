FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "app:app"]
