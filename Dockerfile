# trainer/Dockerfile
# Production-ready Docker image for Vertex AI custom training job

FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-12:latest

# Set working directory
WORKDIR /app

# Install core dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy training source code
COPY . .

# Set environment variables for Vertex AI
ENV PYTHONUNBUFFERED=TRUE
ENV TF_CPP_MIN_LOG_LEVEL=2

# Optional: enable XLA for performance
ENV TF_XLA_FLAGS=--tf_xla_auto_jit=2

# Entry point
ENTRYPOINT ["python", "train.py"]
