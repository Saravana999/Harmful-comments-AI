import streamlit as st
import pickle
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# === Page Configuration ===
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="⚠️", layout="wide", initial_sidebar_state="expanded")

# === Custom CSS for Styling ===
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e2a44 0%, #0f172a 100%);
        color: #e2e8f0;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .card {
        background-color: #1e2a44;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# === Load Models and Vectorizer ===
@st.cache_resource
def load_models_and_vectorizer():
    with open('toxic_comment_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    subtype_model = load_model('subtype_model.h5')
    return logistic_model, vectorizer, subtype_model


# === Prediction Functions ===
def predict_toxicity(texts, model, vectorizer):
    text_vectorized = vectorizer.transform(texts)
    predictions = model.predict_proba(text_vectorized)[:, 1]
    return predictions

def predict_subtypes(texts, model, vectorizer):
    text_vectorized = vectorizer.transform(texts).toarray()
    subtype_probs = model.predict(text_vectorized, verbose=0)
    subtype_labels = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    predicted_subtypes = [
        [label for prob, label in zip(row, subtype_labels) if prob > 0.5] or ['Non-Toxic']
        for row in subtype_probs
    ]
    return predicted_subtypes

# === Create Gauge Chart ===
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ]
        },
        title={'text': "Toxicity Probability (%)"}
    ))
    return fig

# === Main Function ===
def main():
    st.markdown("<h1 style='text-align: center;'>⚠️ Toxic Comment Classifier</h1>", unsafe_allow_html=True)

    # Load models
    logistic_model, vectorizer, subtype_model = load_models_and_vectorizer()

    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=['text', 'probability', 'subtypes'])

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.5, 0.01)

    # Input Area
    st.markdown('<div class="card">', unsafe_allow_html=True)
    comments = st.text_area("Enter comments (one per line):", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction
    if st.button("Analyze Comments"):
        if comments.strip():
            comment_list = [comment.strip() for comment in comments.split('\n') if comment.strip()]
            probabilities = predict_toxicity(comment_list, logistic_model, vectorizer)
            subtypes = predict_subtypes(comment_list, subtype_model, vectorizer)

            predictions = [
                {'text': text, 'probability': prob, 'subtypes': ', '.join(stypes)}
                for text, prob, stypes in zip(comment_list, probabilities, subtypes)
            ]

            new_predictions = pd.DataFrame(predictions)
            st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, new_predictions], ignore_index=True)
        else:
            st.warning("Please enter at least one comment.")

    # Metrics Section
    st.markdown("<h3>🧩 Prediction Overview</h3>", unsafe_allow_html=True)
    total_comments = len(st.session_state.prediction_history)
    toxic_comments = len(st.session_state.prediction_history[st.session_state.prediction_history['probability'] > threshold])
    non_toxic_comments = total_comments - toxic_comments

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Comments", total_comments)
    with col2:
        st.metric("Toxic Comments", toxic_comments)
    with col3:
        st.metric("Non-Toxic Comments", non_toxic_comments)

    # Visuals
    if not st.session_state.prediction_history.empty:
        st.markdown("<h3>📊 Toxicity Visualization</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Gauge Chart
        with col1:
            latest_prob = st.session_state.prediction_history['probability'].iloc[-1]
            st.plotly_chart(create_gauge_chart(latest_prob), use_container_width=True)

        # Pie Chart
        with col2:
            counts = st.session_state.prediction_history['probability'].apply(lambda x: 'Toxic' if x > threshold else 'Non-Toxic').value_counts()
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                title="Toxicity Distribution",
                color=counts.index,
                color_discrete_map={'Toxic': 'red', 'Non-Toxic': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Subtype Chart
        st.markdown("<h3>📊 Subtype Distribution</h3>", unsafe_allow_html=True)
        subtype_counts = {}
        for subtypes in st.session_state.prediction_history['subtypes']:
            for subtype in subtypes.split(', '):
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        subtype_df = pd.DataFrame(list(subtype_counts.items()), columns=['Subtype', 'Count'])
        fig = px.bar(
            subtype_df,
            x='Subtype',
            y='Count',
            labels={'x': 'Subtype', 'y': 'Count'},
            title="Subtype Distribution",
            color='Subtype',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show Predictions
        st.markdown("<h3>📋 Prediction History</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.prediction_history.style.format({'probability': '{:.2%}'}), use_container_width=True)

    # Clear History Button
    if st.button("Clear History"):
        st.session_state.prediction_history = pd.DataFrame(columns=['text', 'probability', 'subtypes'])
        st.rerun()

if __name__ == "__main__":
    main()
