# Dockerfile for AI Cybersecurity Tool API
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed models logs

# Expose port 5001 (updated from 5000 to avoid macOS conflict)
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=api/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run application
CMD ["python", "api/app.py"]
