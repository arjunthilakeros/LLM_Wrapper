# =============================================================================
# Dockerfile for Terminal Chatbot API
# =============================================================================
# Supports both old and new versions via build arguments
#
# Build:
#   docker build --build-arg VERSION=new -t chatbot:new .
#   docker build --build-arg VERSION=old -t chatbot:old .
#
# Run:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY chatbot:new
# =============================================================================

FROM python:3.11-slim

# Build arguments
ARG VERSION=new

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY config.yaml prompts.yaml ./
COPY logger.py exceptions.py validators.py rate_limiter.py ./
COPY config_validator.py api_client.py health.py database.py storage.py ./

# Copy version-specific files
# Note: In production, these would be the migrated files
COPY server.py ./
COPY terminal_chatbot.py ./

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/uploads /app/exports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
