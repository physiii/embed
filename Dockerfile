FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install packaging dependency first (needed for flash-attn)
RUN pip install --no-cache-dir packaging

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"] 