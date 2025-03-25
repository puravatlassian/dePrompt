FROM python:3.10-slim
WORKDIR /app

# Copy dependency lists and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the port as provided by the environment (default to 8080)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn as the production server
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"] 