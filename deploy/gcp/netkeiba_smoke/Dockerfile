FROM python:3.12-slim
WORKDIR /app
COPY smoke.py race_ids.txt ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "smoke.py"]
