"""
Depression Prediction Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
    .risk-high   { background: #fff5f5; border-left: 4px solid #e53e3e; padding: 1.5rem; border-radius: 8px; }
    .risk-low    { background: #f0fff4; border-left: 4px solid #38a169; padding: 1.5rem; border-radius: 8px; }
    .risk-medium { background: #fffbeb; border-left: 4px solid #d69e2e; padding: 1.5rem; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 20px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(BASE, 'model.pkl'))
    encoders = joblib.load(os.path.join(BASE, 'encoders.pkl'))
    scaler   = joblib.load(os.path.join(BASE, 'scaler.pkl'))
    imputer  = joblib.load(os.path.join(BASE, 'imputer.pkl'))
    with open(os.path.join(BASE, 'feature_names.json')) as f:
        feature_names = json.load(f)
    with open(os.path.join(BASE, 'metrics.json')) as f:
        metrics = json.load(f)
    return model, encoders, scaler, imputer, feature_names, metrics

model, encoders, scaler, imputer, feature_names, metrics_data = load_artifacts()

# ─── Preprocessing (mirrors train_model.py) ───────────────────────────────────
SLEEP_MAP = {
    'less than 5 hours': 4, '1-2 hours': 1.5, '2-3 hours': 2.5,
    '3-4 hours': 3.5, '4-5 hours': 4.5, '4-6 hours': 5, '5-6 hours': 5.5,
    '6-7 hours': 6.5, '6-8 hours': 7, '7-8 hours': 7.5, '8 hours': 8,
    '8-9 hours': 8.5, '9-11 hours': 10, '10-11 hours': 10.5,
    'more than 8 hours': 9,
}

def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])

    # Sleep
    sleep_key = df['Sleep Duration'].iloc[0].lower()
    df['Sleep Duration'] = SLEEP_MAP.get(sleep_key, 7.0)

    # Diet encode
    diet_val = df['Dietary Habits'].iloc[0].lower()
    le = encoders['Dietary Habits']
    diet_val = diet_val if diet_val in le.classes_ else 'moderate'
    df['Dietary Habits'] = le.transform([diet_val])[0]

    # Binary
    df['Gender']   = 1 if data['Gender'] == 'Male' else 0
    df['Have you ever had suicidal thoughts ?'] = 1 if data['Have you ever had suicidal thoughts ?'] == 'Yes' else 0
    df['Family History of Mental Illness'] = 1 if data['Family History of Mental Illness'] == 'Yes' else 0
    df['Working Professional or Student']  = 1 if data['Working Professional or Student'] == 'Working Professional' else 0

    # Ensure feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Impute + Scale
    arr = imputer.transform(df)
    arr = scaler.transform(arr)
    return arr

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🔮 Predict", "📊 Model Metrics", "⚖️ Bias Evaluation", "ℹ️ About"])

# ─── Page: Predict ────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Depression Risk Predictor</h1>
        <p>AI-powered mental health risk assessment based on lifestyle and demographic factors</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("⚠️ **Disclaimer:** This tool is for research/educational purposes only. "
            "It is NOT a substitute for professional medical diagnosis.")

    with st.form("prediction_form"):
        st.subheader("👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age    = st.slider("Age", 16, 80, 25)
        with col2:
            role   = st.selectbox("Role", ["Student", "Working Professional"])
        with col3:
            family = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

        st.subheader("📚 Academic / Work Details")
        col4, col5, col6 = st.columns(3)

        is_student = (role == "Student")
        with col4:
            ac_pressure = st.slider("Academic Pressure (1–5)", 0.0, 5.0, 3.0, 0.5,
                                     disabled=not is_student)
            cgpa = st.slider("CGPA (0–10)", 0.0, 10.0, 7.0, 0.1,
                             disabled=not is_student)
        with col5:
            study_sat = st.slider("Study Satisfaction (1–5)", 0.0, 5.0, 3.0, 0.5,
                                  disabled=not is_student)
            wk_pressure = st.slider("Work Pressure (1–5)", 0.0, 5.0, 3.0, 0.5,
                                    disabled=is_student)
        with col6:
            job_sat = st.slider("Job Satisfaction (1–5)", 0.0, 5.0, 3.0, 0.5,
                                disabled=is_student)
            work_hours = st.slider("Work/Study Hours per day", 0.0, 16.0, 8.0, 0.5)

        st.subheader("🏠 Lifestyle")
        col7, col8, col9 = st.columns(3)
        with col7:
            sleep  = st.selectbox("Sleep Duration", list(SLEEP_MAP.keys()), index=6)
        with col8:
            diet   = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        with col9:
            fin_stress = st.slider("Financial Stress (1–5)", 1.0, 5.0, 3.0, 0.5)

        suicidal = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Predict Depression Risk", use_container_width=True)

    if submitted:
        input_data = {
            'Gender': gender, 'Age': age,
            'Working Professional or Student': role,
            'Academic Pressure': ac_pressure if is_student else 0,
            'Work Pressure': wk_pressure if not is_student else 0,
            'CGPA': cgpa if is_student else 0,
            'Study Satisfaction': study_sat if is_student else 0,
            'Job Satisfaction': job_sat if not is_student else 0,
            'Sleep Duration': sleep,
            'Dietary Habits': diet.lower(),
            'Have you ever had suicidal thoughts ?': suicidal,
            'Work/Study Hours': work_hours,
            'Financial Stress': fin_stress,
            'Family History of Mental Illness': family,
        }

        features = preprocess_input(input_data)
        prob = model.predict_proba(features)[0][1]
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 18}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e53e3e" if prob > 0.5 else "#38a169"},
                    'steps': [
                        {'range': [0, 33],  'color': '#c6f6d5'},
                        {'range': [33, 66], 'color': '#fefcbf'},
                        {'range': [66, 100],'color': '#fed7d7'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75, 'value': 50
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if prob > 0.66:
                level, css, emoji = "High Risk", "risk-high", "🔴"
                msg = "The model indicates a **high likelihood of depression**. It is strongly recommended to consult a mental health professional."
            elif prob > 0.33:
                level, css, emoji = "Moderate Risk", "risk-medium", "🟡"
                msg = "The model indicates a **moderate risk of depression**. Consider speaking with a counsellor or trusted person."
            else:
                level, css, emoji = "Low Risk", "risk-low", "🟢"
                msg = "The model indicates a **low risk of depression**. Continue maintaining healthy habits and seek help if needed."

            st.markdown(f"""
            <div class="{css}">
                <h2>{emoji} {level}</h2>
                <p>{msg}</p>
                <p><b>Probability: {prob*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔑 Key Risk Factors")
            risks = []
            if suicidal == "Yes":        risks.append(("Suicidal thoughts", "Very High"))
            if family == "Yes":          risks.append(("Family history", "High"))
            if fin_stress >= 4:          risks.append(("Financial stress", "High"))
            if sleep in ['Less than 5 hours','1-2 hours','2-3 hours','3-4 hours']:
                                         risks.append(("Poor sleep", "Medium"))
            if diet.lower() == "unhealthy": risks.append(("Unhealthy diet", "Medium"))
            if work_hours > 10:          risks.append(("Long work hours", "Medium"))
            if not risks:
                st.success("No major risk flags detected.")
            else:
                for factor, level_r in risks:
                    color = "#e53e3e" if level_r == "Very High" else "#d69e2e" if level_r == "High" else "#667eea"
                    st.markdown(f"- **{factor}** — <span style='color:{color}'>{level_r}</span>", unsafe_allow_html=True)

# ─── Page: Model Metrics ──────────────────────────────────────────────────────
elif page == "📊 Model Metrics":
    st.title("📊 Model Performance Metrics")
    m = metrics_data['overall']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{m['precision']*100:.2f}%")
    col3.metric("Recall",    f"{m['recall']*100:.2f}%")
    col4.metric("F1-Score",  f"{m['f1']*100:.2f}%")

    fig = go.Figure(go.Bar(
        x=list(m.keys()),
        y=[v*100 for v in m.values()],
        marker_color=['#667eea','#764ba2','#f093fb','#f5576c'],
        text=[f"{v*100:.1f}%" for v in m.values()],
        textposition='outside',
    ))
    fig.update_layout(title="Overall Model Metrics", yaxis_title="Score (%)",
                      yaxis=dict(range=[0, 105]), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    | Component | Detail |
    |-----------|--------|
    | Model Type | Multi-Layer Perceptron (Deep Neural Network) |
    | Hidden Layers | 256 → 128 → 64 → 32 neurons |
    | Activation | ReLU |
    | Optimizer | Adam (adaptive learning rate) |
    | Regularization | L2 (alpha=0.001) |
    | Early Stopping | Yes (patience=10) |
    | Training Samples | ~112,560 |
    """)

# ─── Page: Bias Evaluation ───────────────────────────────────────────────────
elif page == "⚖️ Bias Evaluation":
    st.title("⚖️ Fairness & Bias Evaluation")
    st.info("Evaluating model equity across demographic groups.")

    bias = metrics_data['bias']
    for group_name, group_data in bias.items():
        st.subheader(f"📌 {group_name}")
        rows = []
        for subgroup, mvals in group_data.items():
            rows.append({'Group': subgroup, **mvals})
        df_b = pd.DataFrame(rows)
        fig = px.bar(df_b.melt(id_vars='Group', value_vars=['accuracy','precision','recall','f1']),
                     x='Group', y='value', color='variable', barmode='group',
                     title=f"Performance by {group_name}",
                     labels={'value': 'Score', 'variable': 'Metric'},
                     color_discrete_sequence=['#667eea','#764ba2','#f5576c','#f093fb'])
        fig.update_layout(yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_b.set_index('Group').style.format("{:.3f}", subset=['accuracy','precision','recall','f1']),
                     use_container_width=True)

    st.markdown("""
    ### 📝 Fairness Observations
    - **Gender**: Performance is balanced between Male and Female groups (~83% F1 each).
    - **Role**: Working professionals show lower recall for depression cases, likely due to data imbalance.
    - **Age**: Younger groups (<35) are better predicted; older groups (50+) have very few positive cases.
    - **Recommendation**: Collect more diverse data for underrepresented groups to improve equity.
    """)

# ─── Page: About ──────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Depression Risk Prediction
    
    This application uses a **Deep Learning (MLP)** model trained on 140,700 samples to 
    predict the likelihood of depression based on:
    
    - Demographic factors (age, gender)
    - Lifestyle (sleep, diet, work hours)  
    - Academic/professional stress
    - Mental health history
    
    ### Data Pipeline
    ```
    Raw CSV → Cleaning → Encoding → Imputation → Scaling → MLP Model → Prediction
    ```
    
    ### Tech Stack
    | Component | Technology |
    |-----------|-----------|
    | Model | Scikit-learn MLPClassifier |
    | App | Streamlit |
    | Deployment | AWS EC2 / App Runner |
    | Visualisation | Plotly |
    
    ### ⚠️ Ethical Disclaimer
    This tool is intended for **research and educational purposes only**. 
    Depression is a complex medical condition that requires professional diagnosis.
    Always seek help from qualified mental health professionals.
    """)
