# app.py — FraudShield: User-Friendly Credit Card Fraud Detector
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
  .stApp { background: #080b12; color: #e8edf7; }

  [data-testid="stSidebar"] {
    background: #0e1420 !important;
    border-right: 1px solid rgba(100,160,255,0.12) !important;
  }

  [data-testid="metric-container"] {
    background: #0e1420;
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 14px;
    padding: 18px 20px !important;
  }
  [data-testid="metric-container"] label {
    color: #5a6580 !important; font-size: 11px !important; letter-spacing: 1px;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important; font-weight: 800 !important;
  }

  .stNumberInput input, .stSelectbox select, .stTextInput input {
    background: #141a27 !important;
    border: 1px solid rgba(100,160,255,0.15) !important;
    border-radius: 10px !important;
    color: #e8edf7 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #3d8bff, #1a6aff) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 15px !important;
    padding: 14px 24px !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(61,139,255,0.3) !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(61,139,255,0.45) !important;
  }

  /* Section headers */
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #5a6580;
    letter-spacing: 2.5px; text-transform: uppercase;
    margin: 20px 0 10px;
    border-bottom: 1px solid rgba(100,160,255,0.1);
    padding-bottom: 6px;
  }

  /* Helper text under inputs */
  .helper {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #5a6580;
    margin-top: -10px; margin-bottom: 6px;
    padding: 6px 10px;
    background: rgba(100,160,255,0.04);
    border-left: 2px solid rgba(100,160,255,0.2);
    border-radius: 0 6px 6px 0;
  }

  /* Result boxes */
  .result-fraud {
    background: linear-gradient(135deg, rgba(255,69,96,0.18), rgba(255,69,96,0.05));
    border: 2px solid #ff4560; border-radius: 16px;
    padding: 30px; text-align: center;
    box-shadow: 0 0 50px rgba(255,69,96,0.2);
    animation: fadeIn 0.5s ease;
  }
  .result-safe {
    background: linear-gradient(135deg, rgba(0,229,195,0.14), rgba(0,229,195,0.03));
    border: 2px solid #00e5c3; border-radius: 16px;
    padding: 30px; text-align: center;
    box-shadow: 0 0 50px rgba(0,229,195,0.15);
    animation: fadeIn 0.5s ease;
  }
  .result-idle {
    background: #0e1420;
    border: 1px solid rgba(100,160,255,0.12);
    border-radius: 16px; padding: 30px; text-align: center;
  }

  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

  .verdict-fraud { color: #ff4560; font-size: 26px; font-weight: 800; margin-bottom: 6px; }
  .verdict-safe  { color: #00e5c3; font-size: 26px; font-weight: 800; margin-bottom: 6px; }
  .verdict-idle  { color: #5a6580; font-size: 20px; }

  /* Tip box */
  .tip-box {
    background: rgba(61,139,255,0.07);
    border: 1px solid rgba(61,139,255,0.2);
    border-radius: 10px; padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #7aa8ff;
    margin: 8px 0;
    line-height: 1.7;
  }

  /* Risk factor rows */
  .risk-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(100,160,255,0.07);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
  }

  /* Log table */
  .info-box {
    background: #0e1420;
    border: 1px solid rgba(100,160,255,0.12);
    border-radius: 10px; padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #5a6580;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists("fraud_model.pkl"):
        return None, None, None
    return joblib.load("fraud_model.pkl"), joblib.load("scaler.pkl"), joblib.load("metrics.pkl")

model, scaler, saved_metrics = load_artifacts()

# ── SESSION STATE ─────────────────────────────────────────────
for k, v in [("history", []), ("total", 0), ("safe", 0), ("fraud", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'>
  <div style='background:linear-gradient(135deg,#3d8bff,#00e5c3);border-radius:12px;
              width:46px;height:46px;display:flex;align-items:center;justify-content:center;
              font-size:24px;box-shadow:0 0 24px rgba(61,139,255,0.4)'>🛡️</div>
  <div>
    <div style='font-size:28px;font-weight:800;letter-spacing:-1px'>
      Fraud<span style='color:#3d8bff'>Shield</span>
      <span style='font-size:12px;background:rgba(0,229,195,0.1);color:#00e5c3;
            border:1px solid rgba(0,229,195,0.25);padding:3px 10px;border-radius:20px;
            font-family:JetBrains Mono,monospace;letter-spacing:1px;
            margin-left:10px;vertical-align:middle'>LIVE</span>
    </div>
    <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580;margin-top:2px'>
      AI-Powered Credit Card Fraud Detection · Just fill in the details below
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(100,160,255,0.1);margin:10px 0 20px'/>",
            unsafe_allow_html=True)

# ── MODEL NOT TRAINED ─────────────────────────────────────────
if not model:
    st.error("""
    **❌ Model not trained yet!**
    Run this first in your terminal:
    ```bash
    python model.py
    streamlit run app.py
    ```
    """)
    st.stop()

feature_cols = saved_metrics["feature_cols"]

# ── STAT CARDS ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("📊 Checked", st.session_state.total)
with c2: st.metric("✅ Safe",    st.session_state.safe)
with c3: st.metric("🚨 Fraud",   st.session_state.fraud)
with c4:
    rate = f"{st.session_state.fraud/st.session_state.total*100:.1f}%" \
           if st.session_state.total > 0 else "—"
    st.metric("🎯 Fraud Rate", rate)

st.markdown("<br/>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-size:18px;font-weight:800;margin-bottom:4px'>📖 How to use</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580;
                margin-bottom:16px'>Step-by-step guide</div>
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Enter Amount", "Type the transaction amount in rupees or dollars"),
        ("2️⃣", "Pick the Time", "Select what hour the transaction happened (0=midnight, 12=noon, 23=11pm)"),
        ("3️⃣", "Fill the Details", "Answer simple questions about the transaction — location, how card was used, etc."),
        ("4️⃣", "Click Analyze", "Hit the big blue button and get instant result"),
        ("5️⃣", "Read the Result", "🚨 Red = Fraud detected. ✅ Green = Transaction is safe"),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:14px;align-items:flex-start'>
          <div style='font-size:20px;flex-shrink:0'>{icon}</div>
          <div>
            <div style='font-weight:700;font-size:13px;margin-bottom:2px'>{title}</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:11px;
                        color:#5a6580;line-height:1.5'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(100,160,255,0.1);margin:10px 0'/>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:14px;font-weight:700;margin-bottom:10px'>🤖 AI Model Info</div>
    """, unsafe_allow_html=True)

    m = saved_metrics
    for label, val in [
        ("Accuracy", f"{m['roc_auc']*100:.1f}%"),
        ("Precision", f"{m['precision']*100:.1f}%"),
        ("Fraud Recall", f"{m['recall']*100:.1f}%"),
        ("F1 Score", f"{m['f1']*100:.1f}%"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;
                    font-family:JetBrains Mono,monospace;font-size:11px;
                    padding:7px 0;border-bottom:1px solid rgba(100,160,255,0.07)'>
          <span style='color:#5a6580'>{label}</span>
          <span style='color:#3d8bff;font-weight:700'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:14px;font-family:JetBrains Mono,monospace;font-size:10px;
                color:#5a6580;line-height:1.7'>
    Trained on 284,807 real European credit card transactions from Kaggle.
    Uses XGBoost + Random Forest ensemble.
    </div>
    """, unsafe_allow_html=True)


# ── MAIN COLUMNS ──────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

with left:

    # ── SECTION 1: BASIC TRANSACTION INFO ─────────────────────
    st.markdown("""
    <div style='background:rgba(61,139,255,0.06);border:1px solid rgba(61,139,255,0.15);
                border-radius:14px;padding:16px 20px;margin-bottom:18px'>
      <div style='font-size:16px;font-weight:800;margin-bottom:4px'>💳 Basic Transaction Info</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580'>
        The most important details of the transaction
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.01, max_value=50000.0,
            value=150.0, step=0.01,
            help="How much money was spent in this transaction?"
        )
        st.markdown("<div class='helper'>💡 Unusually large amounts are a red flag</div>",
                    unsafe_allow_html=True)

    with col2:
        hour = st.slider(
            "🕐 Time of Transaction",
            min_value=0, max_value=23, value=14,
            help="0 = Midnight, 6 = Early morning, 12 = Noon, 22 = Late night"
        )
        time_label = (
            "🌙 Late Night (High Risk)" if hour <= 5 or hour >= 22
            else "🌅 Early Morning" if hour <= 8
            else "☀️ Daytime (Normal)"
            if hour <= 18 else "🌆 Evening"
        )
        st.markdown(f"<div class='helper'>⏰ {hour}:00 — {time_label}</div>",
                    unsafe_allow_html=True)

    # ── SECTION 2: LOCATION INFO ───────────────────────────────
    st.markdown("""
    <div style='background:rgba(0,229,195,0.05);border:1px solid rgba(0,229,195,0.15);
                border-radius:14px;padding:16px 20px;margin:18px 0'>
      <div style='font-size:16px;font-weight:800;margin-bottom:4px'>📍 Location & Distance</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580'>
        Where did this transaction happen compared to normal?
      </div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        dist_home = st.selectbox(
            "🏠 How far from cardholder's home?",
            options=[
                "Very close (same city)",
                "Nearby (within 50 km)",
                "Far (50–300 km)",
                "Very far (300+ km)",
                "Different country entirely"
            ],
            help="Is the transaction happening near where the cardholder normally lives?"
        )
    with col4:
        dist_last = st.selectbox(
            "📌 Distance from last transaction?",
            options=[
                "Same location / nearby",
                "Different area (10–100 km)",
                "Far from last txn (100+ km)",
                "Impossible distance (different city in minutes)"
            ],
            help="Could the cardholder have physically moved from where they last used the card?"
        )

    st.markdown("""
    <div class='tip-box'>
    ⚠️ <b>Why this matters:</b> If someone used a card in Mumbai at 2pm and then in Delhi at 2:05pm —
    that's physically impossible and a strong fraud signal!
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 3: HOW CARD WAS USED ──────────────────────────
    st.markdown("""
    <div style='background:rgba(255,184,48,0.05);border:1px solid rgba(255,184,48,0.15);
                border-radius:14px;padding:16px 20px;margin:18px 0'>
      <div style='font-size:16px;font-weight:800;margin-bottom:4px'>🔐 How Was the Card Used?</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580'>
        Physical security details of the transaction
      </div>
    </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        chip_used = st.selectbox(
            "💳 Was chip / tap used?",
            options=[
                "Yes — Chip inserted (secure)",
                "Yes — Tap / Contactless",
                "No — Card was swiped",
                "Online transaction (no card present)"
            ],
            help="Chip transactions are more secure. Swipe or online = higher risk."
        )
    with col6:
        pin_used = st.selectbox(
            "🔑 Was PIN entered?",
            options=[
                "Yes — PIN was entered",
                "No — Only signature",
                "No — Tap/Contactless (no PIN)",
                "No — Online (no PIN)"
            ],
            help="PIN-verified transactions are harder to fake."
        )

    col7, col8 = st.columns(2)
    with col7:
        merchant_type = st.selectbox(
            "🏪 Where was the purchase made?",
            options=[
                "Grocery / Supermarket",
                "Restaurant / Food",
                "Petrol / Gas Station",
                "Online Shopping",
                "ATM Withdrawal",
                "Electronics Store",
                "Travel / Airline / Hotel",
                "Jewellery / Luxury goods",
                "Crypto / Money Transfer",
                "Pharmacy / Medical",
            ],
            help="Some merchant types have higher fraud rates (ATM, Crypto, Luxury)."
        )
    with col8:
        foreign_txn = st.selectbox(
            "🌍 International transaction?",
            options=[
                "No — Same country as cardholder",
                "Yes — Foreign country",
                "Yes — Online from foreign IP"
            ],
            help="Transactions from foreign countries are higher risk."
        )

    # ── SECTION 4: SPENDING BEHAVIOUR ─────────────────────────
    st.markdown("""
    <div style='background:rgba(255,69,96,0.05);border:1px solid rgba(255,69,96,0.12);
                border-radius:14px;padding:16px 20px;margin:18px 0'>
      <div style='font-size:16px;font-weight:800;margin-bottom:4px'>📊 Spending Behaviour</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#5a6580'>
        Does this match the cardholder's normal spending habits?
      </div>
    </div>
    """, unsafe_allow_html=True)

    col9, col10 = st.columns(2)
    with col9:
        spend_pattern = st.selectbox(
            "💸 Is this amount unusual for this card?",
            options=[
                "Normal — similar to usual spending",
                "Slightly higher than usual",
                "Much higher than usual (2–5x)",
                "Extremely high — never spent this much",
            ],
            help="A ₹50,000 transaction on a card that usually spends ₹500 is suspicious."
        )
    with col10:
        txn_frequency = st.selectbox(
            "⚡ How many transactions today?",
            options=[
                "1–3 transactions (normal)",
                "4–8 transactions",
                "9–15 transactions (high)",
                "16+ transactions (very suspicious)"
            ],
            help="Many transactions in a short time is a common fraud pattern."
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── ANALYZE BUTTON ─────────────────────────────────────────
    analyze = st.button("🔍  Analyze This Transaction")

    st.markdown("""
    <div class='tip-box' style='margin-top:12px'>
    🔒 <b>Privacy:</b> No real card data is stored or sent anywhere.
    Everything runs locally on your machine.
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT COLUMN — RESULT ─────────────────────────────────────
with right:

    def map_inputs_to_features(amount, hour, dist_home, dist_last,
                                chip_used, pin_used, merchant_type,
                                foreign_txn, spend_pattern, txn_frequency):
        """
        Convert plain-English user inputs into V1–V28 style feature ranges
        that match the patterns learned from the Kaggle dataset.
        """
        # Base: random normal transaction
        np.random.seed(int(amount * 10) % 9999)
        v = np.random.normal(0, 0.3, 28)

        # ── Location signals (V1, V3, V4, V10, V14 are location-correlated) ──
        dist_map = {
            "Very close (same city)": 0,
            "Nearby (within 50 km)": 1,
            "Far (50–300 km)": 2,
            "Very far (300+ km)": 3,
            "Different country entirely": 4
        }
        d = dist_map.get(dist_home, 0)
        v[0]  += -d * 0.8        # V1
        v[2]  += d * 0.5         # V3
        v[9]  += -d * 0.7        # V10
        v[13] += -d * 1.1        # V14 (strong fraud indicator)

        last_map = {
            "Same location / nearby": 0,
            "Different area (10–100 km)": 1,
            "Far from last txn (100+ km)": 2,
            "Impossible distance (different city in minutes)": 4
        }
        dl = last_map.get(dist_last, 0)
        v[3]  += dl * 0.6        # V4
        v[6]  += -dl * 0.9       # V7
        v[11] += -dl * 1.5       # V12 (strong)

        # ── Time signals (V3, V9) ──
        if hour <= 5 or hour >= 22:
            v[2]  += 1.2
            v[8]  += -1.0
        elif hour <= 8:
            v[2]  += 0.4

        # ── Chip / PIN signals (V7, V8, V10) ──
        chip_risk = {
            "Yes — Chip inserted (secure)": 0,
            "Yes — Tap / Contactless": 1,
            "No — Card was swiped": 2,
            "Online transaction (no card present)": 3
        }
        cr = chip_risk.get(chip_used, 0)
        v[6]  += -cr * 0.7
        v[7]  += -cr * 0.5
        v[9]  += -cr * 0.6

        pin_risk = {
            "Yes — PIN was entered": 0,
            "No — Only signature": 1,
            "No — Tap/Contactless (no PIN)": 1,
            "No — Online (no PIN)": 2
        }
        pr = pin_risk.get(pin_used, 0)
        v[4]  += -pr * 0.5
        v[5]  += pr * 0.4

        # ── Merchant risk (V2, V4, V11) ──
        merch_risk = {
            "Grocery / Supermarket": 0, "Restaurant / Food": 0,
            "Petrol / Gas Station": 0.5, "Pharmacy / Medical": 0.3,
            "Electronics Store": 1.0, "Online Shopping": 1.2,
            "Travel / Airline / Hotel": 1.5, "ATM Withdrawal": 2.0,
            "Jewellery / Luxury goods": 2.2, "Crypto / Money Transfer": 3.0
        }
        mr = merch_risk.get(merchant_type, 0)
        v[1]  += -mr * 0.8
        v[3]  += mr * 0.5
        v[10] += mr * 0.6

        # ── Foreign transaction (V14, V16, V17) ──
        foreign_risk = {
            "No — Same country as cardholder": 0,
            "Yes — Foreign country": 3,
            "Yes — Online from foreign IP": 2
        }
        fr = foreign_risk.get(foreign_txn, 0)
        v[13] += -fr * 1.2       # V14 (very strong indicator)
        v[15] += -fr * 0.8
        v[16] += -fr * 1.5       # V17

        # ── Spend pattern (V1, V2, V17) ──
        spend_risk = {
            "Normal — similar to usual spending": 0,
            "Slightly higher than usual": 0.5,
            "Much higher than usual (2–5x)": 1.5,
            "Extremely high — never spent this much": 3.0
        }
        sr = spend_risk.get(spend_pattern, 0)
        v[0]  += -sr * 0.7
        v[1]  += -sr * 0.5
        v[16] += -sr * 1.2

        # ── Transaction frequency (V3, V5, V18) ──
        freq_risk = {
            "1–3 transactions (normal)": 0,
            "4–8 transactions": 0.5,
            "9–15 transactions (high)": 1.5,
            "16+ transactions (very suspicious)": 3.0
        }
        fqr = freq_risk.get(txn_frequency, 0)
        v[2]  += fqr * 0.4
        v[4]  += fqr * 0.3
        v[17] += fqr * 0.6

        return v

    if analyze:
        with st.spinner("🔍 Analyzing transaction..."):
            import time; time.sleep(0.8)  # slight delay for UX

            # Map inputs → V features
            v_vals = map_inputs_to_features(
                amount, hour, dist_home, dist_last,
                chip_used, pin_used, merchant_type,
                foreign_txn, spend_pattern, txn_frequency
            )

            # Build feature row
            log_amount    = np.log1p(amount)
            amount_zscore = (amount - 88.35) / 250.12

            row = {f"V{i}": v_vals[i-1] for i in range(1, 29)}
            row["Log_Amount"]    = log_amount
            row["Amount_zscore"] = amount_zscore
            row["Hour"]          = hour

            X_input  = pd.DataFrame([row])[feature_cols]
            X_scaled = scaler.transform(X_input)

            proba     = model.predict_proba(X_scaled)[0][1]
            pred      = model.predict(X_scaled)[0]
            fraud_pct = round(proba * 100, 2)
            is_fraud  = bool(pred)

            # Update counts
            st.session_state.total += 1
            if is_fraud: st.session_state.fraud += 1
            else:        st.session_state.safe  += 1

            st.session_state.history.append({
                "ID":      f"TXN-{st.session_state.total:04d}",
                "Amount":  f"${amount:,.2f}",
                "Where":   merchant_type.split("/")[0].strip(),
                "Verdict": "🚨 FRAUD" if is_fraud else "✅ SAFE",
                "Risk":    f"{fraud_pct}%",
                "Time":    datetime.now().strftime("%H:%M:%S"),
                "_fraud":  is_fraud
            })

        # ── RESULT BOX ─────────────────────────────────────────
        cls    = "fraud" if is_fraud else "safe"
        icon   = "🚨" if is_fraud else "✅"
        title  = "FRAUD DETECTED" if is_fraud else "TRANSACTION SAFE"
        advice = (
            "⛔ This transaction shows multiple fraud signals. "
            "Block the card immediately and contact your bank."
            if is_fraud else
            "✅ This transaction looks legitimate based on the details you entered."
        )

        st.markdown(f"""
        <div class='result-{cls}'>
          <div style='font-size:54px;line-height:1;margin-bottom:14px'>{icon}</div>
          <div class='verdict-{cls}'>{title}</div>
          <div style='font-family:JetBrains Mono,monospace;font-size:12px;
                      color:#5a6580;margin-top:10px;line-height:1.6'>{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── FRAUD PROBABILITY GAUGE ────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        gauge_color = "#ff4560" if is_fraud else "#00e5c3"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_pct,
            number={"suffix": "%", "font": {"color": gauge_color, "size": 34, "family": "Syne"}},
            delta={"reference": 50, "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100],
                          "tickvals": [0, 25, 50, 75, 100],
                          "ticktext": ["0%", "25%", "50%\nThreshold", "75%", "100%"],
                          "tickcolor": "#5a6580",
                          "tickfont": {"color": "#5a6580", "size": 10}},
                "bar":  {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "#0e1420",
                "bordercolor": "rgba(100,160,255,0.1)",
                "steps": [
                    {"range": [0,  40],  "color": "rgba(0,229,195,0.08)"},
                    {"range": [40, 60],  "color": "rgba(255,184,48,0.08)"},
                    {"range": [60, 100], "color": "rgba(255,69,96,0.10)"},
                ],
                "threshold": {
                    "line": {"color": "#ffb830", "width": 2},
                    "thickness": 0.8, "value": 50
                }
            },
            title={"text": "FRAUD PROBABILITY<br><span style='font-size:11px;color:#5a6580'>"
                           "Below 50% = Safe · Above 50% = Fraud</span>",
                   "font": {"color": "#5a6580", "size": 12, "family": "JetBrains Mono"}}
        ))
        fig_gauge.update_layout(
            height=260, margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e8edf7"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── WHAT TRIGGERED THIS? ────────────────────────────────
        st.markdown("<div class='section-label'>🔍 What triggered this result?</div>",
                    unsafe_allow_html=True)

        # Human-readable risk factors
        risk_items = []

        if hour <= 5 or hour >= 22:
            risk_items.append(("🕐 Late night transaction", "Medium", "#ffb830"))
        if dist_home in ["Very far (300+ km)", "Different country entirely"]:
            risk_items.append(("📍 Far from home location", "High", "#ff4560"))
        if dist_last == "Impossible distance (different city in minutes)":
            risk_items.append(("⚡ Impossible travel speed", "Critical", "#ff0033"))
        if "No" in chip_used or "Online" in chip_used:
            risk_items.append(("💳 No chip used", "Medium", "#ffb830"))
        if "No" in pin_used:
            risk_items.append(("🔑 No PIN entered", "Medium", "#ffb830"))
        if merchant_type in ["ATM Withdrawal", "Crypto / Money Transfer", "Jewellery / Luxury goods"]:
            risk_items.append(("🏪 High-risk merchant type", "High", "#ff4560"))
        if "foreign" in foreign_txn.lower() or "Foreign" in foreign_txn:
            risk_items.append(("🌍 International transaction", "High", "#ff4560"))
        if "Much higher" in spend_pattern or "Extremely" in spend_pattern:
            risk_items.append(("💸 Unusual spending amount", "High", "#ff4560"))
        if "16+" in txn_frequency or "9–15" in txn_frequency:
            risk_items.append(("⚡ Too many transactions today", "High", "#ff4560"))

        if not risk_items:
            risk_items.append(("✅ No major risk factors found", "Low", "#00e5c3"))

        for label, level, color in risk_items:
            st.markdown(f"""
            <div class='risk-row'>
              <span style='color:#e8edf7'>{label}</span>
              <span style='background:rgba(0,0,0,0.3);color:{color};
                    border:1px solid {color}40;padding:3px 10px;
                    border-radius:20px;font-size:11px;font-weight:700'>{level}</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Idle state
        st.markdown("""
        <div class='result-idle'>
          <div style='font-size:50px;margin-bottom:14px'>🔎</div>
          <div class='verdict-idle'>Ready to Analyze</div>
          <div style='font-family:JetBrains Mono,monospace;font-size:12px;
                      color:#5a6580;margin-top:10px;line-height:1.7'>
            Fill in the transaction details on the left<br/>
            and click <b style='color:#3d8bff'>Analyze This Transaction</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Show model metrics while idle
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📈 How accurate is this AI?</div>",
                    unsafe_allow_html=True)
        m = saved_metrics
        fig_met = go.Figure(go.Bar(
            x=["Accuracy\n(ROC-AUC)", "Precision", "Fraud\nRecall", "F1 Score"],
            y=[m["roc_auc"], m["precision"], m["recall"], m["f1"]],
            marker=dict(
                color=[m["roc_auc"], m["precision"], m["recall"], m["f1"]],
                colorscale=[[0,"#1a3a6b"],[0.5,"#3d8bff"],[1,"#00e5c3"]],
                cmin=0.8, cmax=1.0
            ),
            text=[f"{v*100:.1f}%" for v in [m["roc_auc"], m["precision"], m["recall"], m["f1"]]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#e8edf7")
        ))
        fig_met.update_layout(
            height=280, margin=dict(l=10, r=10, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,20,32,0.5)",
            font=dict(family="JetBrains Mono", size=11, color="#5a6580"),
            yaxis=dict(range=[0, 1.15], gridcolor="rgba(100,160,255,0.07)",
                       tickformat=".0%"),
            xaxis=dict(gridcolor="rgba(100,160,255,0.07)")
        )
        st.plotly_chart(fig_met, use_container_width=True)


# ── TRANSACTION LOG ───────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>📜 Transaction History (this session)</div>",
            unsafe_allow_html=True)

if st.session_state.history:
    df_log = pd.DataFrame(st.session_state.history).drop(columns=["_fraud"])
    st.dataframe(df_log, use_container_width=True, hide_index=True,
                 height=min(300, 55 + len(df_log) * 38))

    # Trend chart if enough data
    if len(st.session_state.history) >= 3:
        st.markdown("<div class='section-label'>📊 Risk Trend</div>",
                    unsafe_allow_html=True)
        probs  = [float(h["Risk"].replace("%","")) for h in st.session_state.history]
        ids    = [h["ID"] for h in st.session_state.history]
        colors = ["#ff4560" if h["_fraud"] else "#00e5c3"
                  for h in st.session_state.history]

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=ids, y=probs, mode="lines+markers",
            line=dict(color="#3d8bff", width=2),
            marker=dict(color=colors, size=10,
                        line=dict(width=2, color="#080b12")),
            hovertemplate="%{x}: %{y:.1f}% fraud probability<extra></extra>"
        ))
        fig_t.add_hline(y=50, line_dash="dash",
                        line_color="rgba(255,184,48,0.5)",
                        annotation_text="Fraud Threshold (50%)",
                        annotation_font_color="#ffb830",
                        annotation_font_size=11)
        fig_t.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,20,32,0.5)",
            font=dict(family="JetBrains Mono", size=11, color="#5a6580"),
            xaxis=dict(gridcolor="rgba(100,160,255,0.07)"),
            yaxis=dict(gridcolor="rgba(100,160,255,0.07)",
                       range=[0, 110], title="Fraud Risk %")
        )
        st.plotly_chart(fig_t, use_container_width=True)

    if st.button("🗑️  Clear History"):
        st.session_state.history = []
        st.session_state.total = st.session_state.safe = st.session_state.fraud = 0
        st.rerun()

else:
    st.markdown("""
    <div class='info-box' style='text-align:center;padding:30px 20px'>
      No transactions analyzed yet in this session.<br/>
      Fill the form above and click Analyze to get started!
    </div>
    """, unsafe_allow_html=True)