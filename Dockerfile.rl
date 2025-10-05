FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-2:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install stable-baselines3 gymnasium
COPY . .
ENTRYPOINT ["python", "train_rl.py"]
