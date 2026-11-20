# Python runtime for the inference API
FROM python:3.11-slim

WORKDIR /app

# Install only packages required to serve predictions.
# Training, experiment tracking and test dependencies stay in requirements.txt.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY main.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
