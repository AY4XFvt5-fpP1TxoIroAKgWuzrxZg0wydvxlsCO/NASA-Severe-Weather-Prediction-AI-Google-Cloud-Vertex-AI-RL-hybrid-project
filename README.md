# 🌪️ Severe Weather AI — NASA SpaceApps Edition

### Predicts storm intensity and direction using SAR + reanalysis + RL hybrid modeling.

---

## What It Does
Uses temporal ML (LSTM) trained on SAR and reanalysis data to predict next-step storm intensity and steering angle.  
Reinforcement learning tests sensitivity to SST and wind perturbations.  
Deployed via Vertex AI for scalable inference.

---

## Tech Stack
- **Google Cloud:** BigQuery, Vertex AI, Cloud Storage  
- **ML Frameworks:** TensorFlow 2.x, Stable-Baselines3 (RL)  
- **Languages:** Python 3.10, SQL  
- **Data:** NASA SAR imagery, NOAA reanalysis, IBTrACS storms  

---

## How It Works
1. Prepares spatiotemporal sequences in BigQuery  
2. Trains LSTM model on storm evolution  
3. Deploys to Vertex AI for online predictions  
4. RL explores causal sensitivities to environmental changes  

---

## Run Locally
bash
export GOOGLE_APPLICATION_CREDENTIALS=~/key.json
export BQ_SEQUENCE_TABLE=project.dataset.sequence_features
python trainer/train.py
python rl/train_rl.py
