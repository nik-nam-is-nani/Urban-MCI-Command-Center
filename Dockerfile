# Urban MCI Environment Docker Image
# For HuggingFace Spaces deployment

# Stage 1: Build React dashboard
FROM node:18-alpine AS dashboard-builder

WORKDIR /app/dashboard

# Copy dashboard files
COPY dashboard/package.json ./
COPY dashboard/package-lock.json* ./
COPY dashboard/public ./public
COPY dashboard/src ./src

# Install dependencies and build
RUN npm ci
RUN npm run build

# Stage 2: Python Flask server with built dashboard
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

# Copy built dashboard from builder stage
COPY --from=dashboard-builder /app/dashboard/build ./dashboard/build
COPY --from=dashboard-builder /app/dashboard/public ./dashboard/public

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for API
EXPOSE 7860

# Run API server
CMD ["python", "app.py"]
