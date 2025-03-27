# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GUNICORN_WORKERS=1
ENV GUNICORN_THREADS=2
ENV GUNICORN_MAX_REQUESTS=500
ENV GUNICORN_MAX_REQUESTS_JITTER=50
ENV GUNICORN_TIMEOUT=120
ENV GUNICORN_WORKER_CLASS="gthread"

# Command to run the application with optimized settings
CMD gunicorn app:app \
    --bind 0.0.0.0:8000 \
    --workers=${GUNICORN_WORKERS} \
    --threads=${GUNICORN_THREADS} \
    --worker-class=${GUNICORN_WORKER_CLASS} \
    --max-requests=${GUNICORN_MAX_REQUESTS} \
    --max-requests-jitter=${GUNICORN_MAX_REQUESTS_JITTER} \
    --timeout=${GUNICORN_TIMEOUT} 