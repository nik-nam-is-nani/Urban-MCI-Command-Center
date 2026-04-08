# Urban MCI Environment Docker Image
# For HuggingFace Spaces deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY urban_mci_env.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Copy dashboard if it exists
COPY dashboard/ dashboard/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for API
EXPOSE 8000

# Run API server (NOT inference script)
CMD ["python", "app.py"]
