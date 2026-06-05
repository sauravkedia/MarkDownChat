# ---------- Stage 1: Build ----------
# Use a lightweight Python image
FROM python:3.13-slim AS builder

# Set environment for virtual environment location
ENV VENV_PATH="/opt/venv"

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.13 -m venv $VENV_PATH

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in virtual environment
RUN $VENV_PATH/bin/pip install --upgrade pip && \
    $VENV_PATH/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.13-slim AS runtime

# Set environment for virtual environment location
ENV VIRTUAL_ENV="/opt/venv"

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv \
    PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . /app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the FastAPI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" || exit 1

# Run the FastAPI application
CMD ["python", "chat_document.py", "--api"]