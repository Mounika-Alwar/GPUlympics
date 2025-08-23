import streamlit as st 
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title = "GPUlympics",layout = "wide")

st.title("GPUlympics")

st.markdown("Interactive System that benchmarks and predicts GPU efficiency for large scale AI training. Presented in both Gamified+Datascientist Mode")

mode = st.radio("Select Mode: ",["Gamification Mode","DataScientist Mode"])

API_URL = "https://gpulympics-1.onrender.com/predict"

df = pd.read_csv("data/synthetic_gpu_training.csv")
df_avg = df.groupby(['gpu_type', 'model_size', 'batch_size', 'seq_length', 'learning_rate'])[['training_time_hrs', 'energy_kwh', 'efficiency_tok_per_watt']].mean().reset_index()

if mode == "Gamification Mode":
    st.header("Explore GPU performance interactively!")
    st.write(
        """
        You are requested to choose batch size, learning rate, sequence length, model_size 
        and run_id (optional) and this will return the performance of each GPU.
        The app will tell you the fastest GPU, the greenest GPU, and the most efficient GPU.
        """
    )

    # Sidebar inputs
    st.sidebar.header("🛠️ Choose your configuration")
    model_size = st.sidebar.selectbox("Model Size", ["7B", "70B", "405B"])
    batch_size = st.sidebar.selectbox("Batch Size", [256, 512, 1024, 2048, 4096, 8192])
    learning_rate = st.sidebar.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4])
    seq_length = st.sidebar.selectbox("Sequence Length", [512, 1024, 2048, 4096])
    run_id = st.sidebar.number_input("Run ID (optional)", min_value=0, value=1)

    if st.sidebar.button("Predict"):
        # Prepare payload
        payload = {
                "model_size": model_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seq_length": seq_length,
                "run_id": run_id
                }
        with st.spinner("Crunching numbers..."):
            response = requests.post(API_URL, json=payload)
        # Call backend API
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            all_gpu = pd.DataFrame(data['gpu_comparison'])

            col1, col2, col3 = st.columns(3)
            col1.metric("**Fastest GPU**", data['best_gpus']['fastest'])
            col2.metric("**Greenest GPU**", data['best_gpus']['greenest'])
            col3.metric("**Most Efficient GPU**", data['best_gpus']['most_efficient'])
            
            # Training Time
            st.subheader("⏱️ GPU vs Training Time")
            fig1 = px.bar(all_gpu, x='gpu_type', y='training_time_hrs',color='gpu_type')
            st.plotly_chart(fig1, use_container_width=True)
            st.write(f"**{data['best_gpus']['fastest']} is the Fastest GPU for chosen configuration**")

            # Energy Consumption
            st.subheader("🔋 GPU vs Energy Consumption")
            fig2 = px.bar(all_gpu, x='gpu_type', y='energy_kwh',color='gpu_type')
            st.plotly_chart(fig2, use_container_width=True)
            st.write(f"**{data['best_gpus']['greenest']} is the Greenest GPU for chosen configuration**")

            # Efficiency
            st.subheader("⚡ GPU vs Efficiency")
            fig3 = px.bar(all_gpu, x='gpu_type', y='efficiency_tok_per_watt',color='gpu_type')
            st.plotly_chart(fig3, use_container_width=True)
            st.write(f"**{data['best_gpus']['most_efficient']} is the Most Efficient GPU for chosen configuration**")
        else:
            st.error(f"Error: {response.text}")
else:
    st.sidebar.header("📌 DataScientist Mode Navigation")

    section = st.sidebar.selectbox("Jump to Section", [
        "Introduction",
        "Findings and Insights",
        "Model Explanation",
        "Limitations",
        "Future Work"
    ])



    if section == "Introduction":
        st.subheader("Introduction")
        st.write("""
            The idea behind this project is pretty simple: we wanted to understand how different GPUs behave when training a model. Not just in terms of speed, but also how much energy they consume and how efficient they are overall. So we built a synthetic dataset based on rule-based insights from **NVIDIA**’s three most popular GPUs — **A100**, **H100**, and **GB200** — which are widely used in the deep learning world.

            Using this dataset, we trained a model that takes in your configuration (like batch size, learning rate, model size, etc.) and predicts how each GPU would perform. Then, it tells you:

            🏎️ Which GPU is the **fastest** (least training time)  
            🌱 Which one is the **greenest** (least energy consumed)  
            ⚡ Which one is the **most efficient** (best performance per watt)

            To make this exploration engaging, the project introduces a gamified environment where users can tweak configurations and observe which GPU “wins” under different scenarios.
            """)
    if section == "Findings and Insights":
        st.subheader("Findings and Insights")
        st.markdown("##### 1. GPU Type Sets the Pace for Traning Speed")
        st.write("Latest GPUs like **GB200** cut training time significantly compared to **A100** due to higher throughput, making advanced hardware a must for speed.")
        
        fig = px.box(
            df_avg,
            x='gpu_type',
            y='training_time_hrs',
            color='model_size',
            facet_col='seq_length',
            height=500,
            width=1500,
            title="Training Time by GPU Type and Model Size Across Sequence Lengths",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['batch_size', 'learning_rate']
        )
        fig.update_yaxes(type='log', title="Training Time (Hours)")
        fig.update_layout(
            xaxis_title="GPU Type",
            showlegend=True,
            font=dict(size=14),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 2. Batch Size is a Double Win for Time and Energy")
        st.write("Smaller batch sizes spike both training time and energy consumption; cranking up batch size (e.g., to 8192) reduces both, highlighting its key role in optimization.")
        
        fig = px.scatter(
            df_avg,
            x='batch_size',
            y='training_time_hrs',
            size='energy_kwh',
            color='model_size',
            height=500,
            width=800,
            title="Training Time vs. Batch Size (Sized by Energy Consumption)",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['gpu_type']
        )
        fig.update_xaxes(type='log', title="Batch Size")
        fig.update_yaxes(type='log', title="Training Time (Hours)")
        fig.update_layout(
            showlegend=True,
            font=dict(size=14),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 3. Long Sequences Drive Up Time and Energy Costs")
        st.write("As seq_length grows (from 512 to 4096), training time and energy consumption rise sharply, worsened by non-optimal learning_rate.")
        
        fig = px.scatter(
            df_avg,
            x='seq_length',
            y='training_time_hrs',
            size='energy_kwh',
            color='learning_rate',
            height=500,
            width=800,
            title="Training Time vs. Sequence Length (Sized by Energy Consumption)",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=['gpu_type', 'model_size', 'batch_size']
        )
        fig.update_xaxes(type='log', title="Sequence Length")
        fig.update_yaxes(type='log', title="Training Time (Hours)")
        fig.update_layout(
            showlegend=True,
            font=dict(size=14),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 4. Energy Consumption Stays Consistent Across GPUs")
        st.write("For a given seq_length, energy use is roughly similar among A100, H100, and GB200, with minimal differences—focusing the green choice on other factors like model size and learning_rate")
        
        fig = px.box(
            df_avg,
            x='gpu_type',
            y='energy_kwh',
            color='model_size',
            facet_col='seq_length',
            height=500,
            width=1500,
            title="Energy Consumption by GPU Type Across Sequence Lengths",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['batch_size', 'learning_rate']
        )
        fig.update_yaxes(type='log', title="Energy Consumption (kWh)")
        fig.update_layout(
            xaxis_title="GPU Type",
            showlegend=True,
            font=dict(size=14),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 5. Efficiency Peaks at Low Seq_Length")
        st.write("The shortest seq_length (512) delivers the highest efficiency (tokens/watt) for all GPUs, especially for 7B models on GB200, dropping as seq_length increases.")
        
        fig = px.scatter(
            df_avg,
            x='seq_length',
            y='efficiency_tok_per_watt',
            color='model_size',
            height=500,
            width=800,
            title="Efficiency (Tokens/Watt) vs. Sequence Length",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=['gpu_type', 'batch_size', 'learning_rate']
        )
        fig.update_xaxes(type='log', title="Sequence Length")
        fig.update_yaxes(title="Efficiency (Tokens/Watt)")
        fig.update_layout(
            showlegend=True,
            font=dict(size=14),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    if section == "Model Explanation":
        st.subheader("Model Explanation")
        st.write("""
        This model predicts three key metrics for training large language models:
        - **Training time (hrs)**
        - **Energy consumption (kWh)**
        - **Efficiency (tokens per watt)**

        Predictions are based on user-defined hyperparameters and GPU type.
        """)

        st.write("#### What It Learns")
        st.write("""
        Trained a multi-output regression model using a Random Forest ensemble wrapped in `MultiOutputRegressor`.  
        It estimates normalized performance metrics across different GPU types using:
        - Model size (ordinal mapped: 7B → 0, 70B → 1, 405B → 2)
        - Batch size
        - Learning rate
        - Sequence length
        - Run ID
        - GPU type (one-hot encoded)

        To reduce GPU-specific bias, each target variable is normalized by subtracting the mean within its GPU group.
        """)

        st.write("#### Preprocessing Pipeline")
        st.write("""
        - **Model Size Mapping**: Converts string labels to ordinal values  
        - **GPU Encoding**: One-hot encodes GPU type  
        - **ColumnTransformer + Pipeline**: Ensures modular, reproducible preprocessing
        """)

        st.write("#### Model Training")
        st.write("""
        - Data split: 80/20 train-test  
        - Base model: Random Forest Regressor  
        - Wrapper: MultiOutputRegressor for simultaneous prediction of all three targets  
        - Initial test MSE: **5405.80**
        """)

        st.write("#### Hyperparameter Tuning")
        st.write("""
        Performed randomized search over 100 fits using 5-fold cross-validation.  
        **Best parameters:**
        """)
        st.code("""
        {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': None
        }
        """, language='python')
        st.write("Best CV MSE: **4979.02**")

        st.write("#### Prediction Logic")
        st.write("""
        Given a set of hyperparameters, the model simulates training outcomes across three GPU types:  
        **A100**, **H100**, and **GB200**. It then highlights:
        - **Fastest GPU** → lowest predicted training time  
        - **Greenest GPU** → lowest predicted energy usage  
        - **Most Efficient GPU** → highest tokens per watt  

        This helps users make informed decisions about compute-resource allocation for LLM training.
        """)



    if section == "Limitations":
        st.subheader("Limitations")
        st.write("""
        While the model offers valuable insights into GPU performance for LLM training, it has a few important limitations:
        - **Synthetic Data**: The training data is synthetically generated or simulated, which may not fully capture the variability and noise present in real-world training environments.
        - **Limited GPU Scope**: Predictions are restricted to three GPU types (A100, H100, GB200). Emerging hardware or hybrid setups are not yet modeled.
        - **Static Assumptions**: The model assumes fixed environmental conditions and does not account for dynamic factors like thermal throttling, power fluctuations, or workload interference.
        - **Normalization Bias**: While normalization reduces GPU-specific bias, it may also mask edge-case behaviors or outlier performance patterns.
        """)

    if section == "Future Work":
        st.subheader("Future Work")
        st.write("""
        To improve accuracy and real-world applicability, future iterations of this project could include:
        - **Real-World Data Ingestion**: Integrate telemetry logs or benchmark datasets from actual LLM training runs to replace or augment synthetic data.
        - **Realtime Adaptation**: Enable the model to update dynamically based on live training metrics, allowing for adaptive predictions and alerts.
        - **Expanded GPU Coverage**: Add support for newer GPUs and multi-GPU configurations to reflect evolving hardware landscapes.
        - **Deployment Monitoring**: Build a dashboard layer that tracks actual vs. predicted metrics during model deployment, enabling feedback loops and continuous learning.
        - **Explainability Layer**: Incorporate SHAP or feature importance visualizations to help users understand which hyperparameters most influence each metric.
        """)
            
