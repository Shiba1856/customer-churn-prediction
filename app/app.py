import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns = joblib.load('models/model_columns.pkl')

st.set_page_config(page_title="Churn Intelligence", page_icon="🪐", layout="wide")

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(124, 58, 237, 0.35), transparent),
        linear-gradient(180deg, #05060f 0%, #0a0b1a 40%, #0d0e1f 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header { visibility: hidden; }

/* ---------- HERO ---------- */
.hero-wrap { text-align: center; padding: 60px 20px 40px 20px; }
.hero-title {
    font-size: 64px; font-weight: 900; line-height: 1.08; color: #f8fafc;
    letter-spacing: -1.5px; margin-bottom: 16px;
}
.hero-title span {
    background: linear-gradient(90deg, #a78bfa, #7c3aed, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 19px; color: #94a3b8; max-width: 620px; margin: 0 auto; font-weight: 400; }

/* ---------- GLASS CARDS ---------- */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px; padding: 32px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.glass-card:hover { border: 1px solid rgba(124,58,237,0.35); transition: 0.3s; }

.kpi-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(255,255,255,0.02));
    border: 1px solid rgba(124,58,237,0.25); border-radius: 18px; padding: 22px 26px;
}
.kpi-label { color: #a78bfa; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { color: #f8fafc; font-size: 34px; font-weight: 800; margin-top: 6px; }

.badge-high {
    background: linear-gradient(90deg, #7f1d1d, #991b1b); color: #fecaca;
    padding: 8px 18px; border-radius: 999px; font-weight: 700; font-size: 13px;
    box-shadow: 0 0 20px rgba(239,68,68,0.3);
}
.badge-low {
    background: linear-gradient(90deg, #064e3b, #065f46); color: #a7f3d0;
    padding: 8px 18px; border-radius: 999px; font-weight: 700; font-size: 13px;
    box-shadow: 0 0 20px rgba(16,185,129,0.3);
}

.prob-number { font-size: 52px; font-weight: 900; color: #f8fafc; margin: 8px 0; }

/* ---------- RADAR ANIMATION (empty state) ---------- */
.radar-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 60px 20px; }
.radar {
    position: relative; width: 140px; height: 140px; margin-bottom: 28px;
}
.radar-ring {
    position: absolute; border-radius: 50%; border: 2px solid rgba(168,85,247,0.5);
    top: 50%; left: 50%; transform: translate(-50%,-50%);
    animation: radar-pulse 3s cubic-bezier(0.4,0,0.2,1) infinite;
}
.radar-ring:nth-child(1) { width: 40px; height: 40px; animation-delay: 0s; }
.radar-ring:nth-child(2) { width: 40px; height: 40px; animation-delay: 1s; }
.radar-ring:nth-child(3) { width: 40px; height: 40px; animation-delay: 2s; }
.radar-core {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    width: 40px; height: 40px; border-radius: 50%;
    background: radial-gradient(circle, #c084fc, #7c3aed);
    box-shadow: 0 0 30px rgba(168,85,247,0.8);
}
@keyframes radar-pulse {
    0% { width: 40px; height: 40px; opacity: 0.8; }
    100% { width: 140px; height: 140px; opacity: 0; }
}
.radar-text-title { font-size: 20px; font-weight: 800; color: #f8fafc; margin-bottom: 8px; }
.radar-text-sub { color: #94a3b8; font-size: 14px; text-align: center; max-width: 320px; }

/* Inputs */
section[data-testid="stSidebar"] { background: rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.06); }
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #a855f7); color: white; border: none;
    border-radius: 12px; font-weight: 700; padding: 12px 0; box-shadow: 0 0 24px rgba(124,58,237,0.4);
}
.stButton>button:hover { box-shadow: 0 0 32px rgba(124,58,237,0.6); }

h3, h4 { color: #f1f5f9 !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Know who's leaving<br><span>before they do.</span></div>
    <div class="hero-sub">A Random Forest model trained on 7,000+ customer records — predicting churn risk from tenure, billing, and contract behavior in real time.</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯  Predict a Customer", "📁  Batch Analysis"])

def build_input(tenure, monthly_charges, total_charges, senior, contract, internet,
                 payment, partner, dependents, paperless):
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    for col, val in [('tenure', tenure), ('MonthlyCharges', monthly_charges),
                      ('TotalCharges', total_charges), ('SeniorCitizen', senior)]:
        if col in input_df.columns:
            input_df[col] = val
    for col, val in [
        (f'Contract_{contract}', True), (f'InternetService_{internet}', True),
        (f'PaymentMethod_{payment}', True), ('Partner_Yes', partner),
        ('Dependents_Yes', dependents), ('PaperlessBilling_Yes', paperless),
    ]:
        if col in input_df.columns and val:
            input_df[col] = 1
    return input_df

# ================= TAB 1 =================
with tab1:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Customer Profile")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        c1, c2 = st.columns(2)
        senior = c1.checkbox("Senior Citizen")
        partner = c1.checkbox("Has Partner")
        dependents = c2.checkbox("Has Dependents")
        paperless = c2.checkbox("Paperless Billing", value=True)
        predict_btn = st.button("Predict Churn Risk →", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if predict_btn:
            input_df = build_input(tenure, monthly_charges, total_charges, 1 if senior else 0,
                                     contract, internet, payment, partner, dependents, paperless)
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            badge = '<span class="badge-high">⚠ HIGH RISK</span>' if pred == 1 else '<span class="badge-low">✓ LOW RISK</span>'
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f'<div class="prob-number">{prob:.1%}</div>', unsafe_allow_html=True)
            st.caption("Estimated churn probability")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={'suffix': '%', 'font': {'size': 1, 'color': 'rgba(0,0,0,0)'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                       'bar': {'color': "#a855f7" if prob <= 0.5 else "#ef4444"},
                       'bgcolor': 'rgba(255,255,255,0.03)', 'borderwidth': 0,
                       'steps': [{'range': [0, 40], 'color': 'rgba(16,185,129,0.15)'},
                                 {'range': [40, 70], 'color': 'rgba(245,158,11,0.15)'},
                                 {'range': [70, 100], 'color': 'rgba(239,68,68,0.15)'}]}
            ))
            fig.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10),
                               paper_bgcolor='rgba(0,0,0,0)', font={'color': '#f8fafc'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Top factors")
            importances = pd.Series(model.feature_importances_, index=columns)
            top = importances.sort_values(ascending=False).head(6)
            fig2 = px.bar(x=top.values, y=top.index, orientation='h')
            fig2.update_traces(marker_color='#a855f7')
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#cbd5e1'}, showlegend=False,
                                yaxis={'autorange': 'reversed', 'gridcolor': 'rgba(255,255,255,0.05)'},
                                xaxis={'gridcolor': 'rgba(255,255,255,0.05)'})
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <div class="radar-wrap">
                    <div class="radar">
                        <div class="radar-ring"></div>
                        <div class="radar-ring"></div>
                        <div class="radar-ring"></div>
                        <div class="radar-core"></div>
                    </div>
                    <div class="radar-text-title">Scanning for risk signals</div>
                    <div class="radar-text-sub">Set the customer profile on the left, then hit Predict — the model will score churn risk in real time.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Upload a customer list (CSV)")
    st.caption("Same schema as the Telco dataset — we'll score every row instantly.")
    file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if file:
        raw = pd.read_csv(file)
        raw_proc = raw.copy()
        if 'customerID' in raw_proc.columns:
            raw_proc = raw_proc.drop('customerID', axis=1)
        if 'TotalCharges' in raw_proc.columns:
            raw_proc['TotalCharges'] = pd.to_numeric(raw_proc['TotalCharges'], errors='coerce').fillna(0)
        if 'Churn' in raw_proc.columns:
            raw_proc = raw_proc.drop('Churn', axis=1)

        cat_cols = raw_proc.select_dtypes(include='object').columns.tolist()
        encoded = pd.get_dummies(raw_proc, columns=cat_cols, drop_first=True).reindex(columns=columns, fill_value=0)

        preds = model.predict(encoded)
        probs = model.predict_proba(encoded)[:, 1]
        results = raw.copy()
        results['Churn_Probability'] = (probs * 100).round(1)
        results['Prediction'] = np.where(preds == 1, 'At Risk', 'Safe')

        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        for col, label, value in [(k1, "Total Customers", f"{len(results):,}"),
                                    (k2, "At-Risk", f"{(preds==1).sum():,}"),
                                    (k3, "Avg Risk", f"{probs.mean()*100:.1f}%")]:
            col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                          f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(results.sort_values('Churn_Probability', ascending=False), use_container_width=True, height=380)
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Results", csv, "churn_predictions.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)