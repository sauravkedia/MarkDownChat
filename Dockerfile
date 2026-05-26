# ---------- Stage 1: Build ----------
# Use a lightweight Python image
FROM python:3.12 AS builder

# Set environment for virtual environment location
ENV VENV_PATH="/opt/venv"

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update 
RUN apt-get install -y --no-install-recommends build-essential 
RUN rm -rf /var/lib/apt/lists/*

# Install dependencies in a virtual environment (editable optional)
RUN python3.12 -m venv $VENV_PATH

# Copy the project files
COPY requirement.txt .

RUN $VENV_PATH/bin/pip install --upgrade pip 
RUN $VENV_PATH/bin/pip install -r requirement.txt
# RUN /opt/venv/bin/activate
# RUN pip install --no-cache-dir -r requirement.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.12 AS runtime

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy the source code
COPY . /app

# Install Ollama inside the container (optional - but better to use the ollama service)
RUN curl -fsSL https://ollama.ai/install.sh | sh
RUN ollama pull qwen2.5

# Expose the port (if running an API)
EXPOSE 8000

# Define the command to run the AI agent
CMD ["python", "chat_document.py"]