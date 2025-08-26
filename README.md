# GPUlympics

GPUlympics is an interactive system that benchmarks and predicts GPU efficiency for large-scale AI training. It supports both a **gamified exploration mode** and a **data scientist mode** with interactive insights.

## Features
- Predicts the **fastest, greenest, and most efficient GPU** for a given configuration
- Built on a **synthetic dataset** simulating NVIDIA GPUs (A100, H100, GB200)
- Uses **Random Forest with MultiOutputRegressor**, hyper-tuned for best predictions
- Interactive **Streamlit frontend** with **Plotly visualizations**
- FastAPI backend serving predictions via API

## Project Structure
- `backend/` : FastAPI code and preprocessing pipeline
- `frontend/` : Streamlit app
- `data/` : Synthetic dataset used for training

Live app: https://mounika-alwar-gpulympics-frontendfrontend-j3d29g.streamlit.app/
