FROM python:3.12-slim

# Faster Python, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime deps (certs, timezone); no build tools needed for current reqs
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure writable dirs
RUN mkdir -p /app/logs

# By default runs one-off summarization. Pass envs via --env-file or -e
CMD ["python", "summarizer.py"]


