#!/bin/bash
# Deploy TensorFlow model to Vertex AI

PROJECT_ID="your-gcp-project"
REGION="us-central1"
MODEL_DIR="gs://your-bucket/models/severe-weather"
MODEL_NAME="severe-weather-lstm"

gcloud ai models upload \
  --region=$REGION \
  --display-name=$MODEL_NAME \
  --artifact-uri=$MODEL_DIR \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest

ENDPOINT_NAME="${MODEL_NAME}-endpoint"

gcloud ai endpoints create \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME
