# ===============================================================
# 🌌 Exoplanet Discovery Hub — AI-Powered TESS Planet Explorer
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from lightkurve import search_lightcurve
import io
import os
import openai
from datetime import datetime

# ===============================================================
# 1️⃣ APP CONFIG
# ===============================================================
st.set_page_config(page_title="Exoplanet Discovery Hub", page_icon="🪐", layout="wide")

st.title("🔭 Exoplanet Discovery Hub")
st.markdown("### Discover, analyze, and understand planets beyond our solar system — using real NASA TESS data and AI insights.")

# Load your OpenAI key from Streamlit secrets
# Ensure you have 'OPENAI_API_KEY' set up in your Streamlit secrets file (or environment variables)
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)

# ===============================================================
# 2️⃣ SIDEBAR
# ===============================================================
st.sidebar.header("🔧 Controls")
mode = st.sidebar.radio("Choose a mode:", ["Analyze TESS TIC ID", "Upload Light Curve", "Saved Planet Profiles", "AI Planet Assistant"])

# Storage for discovered planets
if "planet_db" not in st.session_state:
    st.session_state.planet_db = []

# ===============================================================
# 3️⃣ HELPER FUNCTIONS
# ===============================================================

def fetch_tess_data(tic_id):
    """Fetch TESS light curve data for a given TIC ID."""
    try:
        # FIX APPLIED: Added 'cache=False' to bypass the corrupt file in the lightkurve cache.
        lc_collection = search_lightcurve(f"TIC {tic_id}", mission="TESS").download_all(cache=False)
        if lc_collection is None:
            return None
        lc = lc_collection.stitch().remove_nans().normalize()
        return lc.to_pandas()
    except Exception as e:
        st.error(f"⚠️ Could not fetch data for TIC {tic_id}: {e}")
        return None


def detect_transits(df):
    """Simple dip detection (mock algorithm for demonstration)."""
    # Ensure necessary columns exist before proceeding
    if 'flux' not in df.columns or 'time' not in df.columns:
        st.error("Uploaded data must contain 'time' and 'flux' columns.")
        return None
        
    df["rolling_flux"] = df["flux"].rolling(window=15, center=True).mean()
    df["dip"] = df["flux"] < (df["rolling_flux"].mean() - 2 * df["rolling_flux"].std())
    dips = df[df["dip"]]
    if dips.empty:
        return None
    # Calculate depth relative to the median flux to handle normalization edge cases
    median_flux = df["flux"].median()
    dip_mean_flux = dips["flux"].mean()
    depth = (median_flux - dip_mean_flux) / median_flux if median_flux > 0 else 0
    period_guess = np.mean(np.diff(dips["time"].values)) if len(dips) > 1 else np.nan
    return {"depth": depth, "period": period_guess, "num_dips": len(dips)}


def generate_ai_summary(tic_id, metrics):
    """Use GPT to generate an explanation."""
    if not openai.api_key:
        return "🔒 OpenAI API key missing. Please add it in Streamlit secrets."
    
    # Ensure all metric keys are present before formatting the prompt
    if not all(k in metrics for k in ['depth', 'period', 'num_dips']):
        return "Error: Incomplete metrics data provided to AI summary function."

    prompt = f"""
    You are an astrophysics expert. Explain in simple terms what these values mean for a possible exoplanet:
    - TIC ID: {tic_id}
    - Transit depth: {metrics['depth']:.5f} (relative flux change)
    - Period (days): {metrics['period']:.2f}
    - Number of transits detected: {metrics['num_dips']}
    Mention what kind of planet this could be (e.g., Hot Jupiter, Earth-sized) and whether the period suggests it might be in the habitable zone.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI explanation unavailable: {e}"


def save_planet_profile(tic_id, metrics, ai_summary):
    """Save discovered planet info."""
    st.session_state.planet_db.append({
        "TIC ID": tic_id,
        "Transit Depth": round(metrics["depth"], 5),
        "Estimated Period (days)": round(metrics["period"], 2) if metrics.get("period") else "N/A",
        "Number of Dips": metrics.get("num_dips", 0),
        "AI Summary": ai_summary,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })


# ===============================================================
# 4️⃣ MODE: ANALYZE TESS TIC ID
# ===============================================================
if mode == "Analyze TESS TIC ID":
    st.subheader("🪐 Analyze a Star from NASA TESS Database")

    tic_id = st.text_input("Enter TIC ID (e.g., 307210830):")
    if st.button("🔍 Analyze TIC"):
        if tic_id:
            st.info(f"Fetching TESS data for TIC {tic_id}...")
            data = fetch_tess_data(tic_id)

            if data is not None:
                st.success("✅ Data fetched successfully!")
                
                # Using Matplotlib for a cleaner, labelled light curve plot
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data["time"], data["flux"], marker='.', linestyle='none', alpha=0.5, markersize=2)
                ax.set_xlabel("Time (BJD)")
                ax.set_ylabel("Normalized Flux")
                ax.set_title(f"Light Curve for TIC {tic_id}")
                st.pyplot(fig) 
                # CRITICAL FIX 1: Close the Matplotlib figure immediately to prevent memory leaks/crashes
                plt.close(fig)

                st.write("Detecting possible transit events...")
                metrics = detect_transits(data)
                if metrics:
                    st.success("🪐 Possible planet-like signals detected!")
                    st.write(metrics)

                    ai_summary = generate_ai_summary(tic_id, metrics)
                    st.markdown("### 🤖 AI Explanation")
                    st.info(ai_summary)

                    if st.button("💾 Save Planet Profile"):
                        save_planet_profile(tic_id, metrics, ai_summary)
                        st.success("Planet profile saved!")
                else:
                    st.warning("No significant transit patterns found.")
            else:
                st.error("Failed to retrieve TESS data.")
        else:
            st.warning("Please enter a valid TIC ID.")

# ===============================================================
# 5️⃣ MODE: UPLOAD LIGHT CURVE
# ===============================================================
elif mode == "Upload Light Curve":
    st.subheader("📂 Upload a Custom Light Curve File (.csv)")

    uploaded_file = st.file_uploader("Upload a CSV file with columns `time` and `flux`", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Using Matplotlib for a cleaner, labelled light curve plot
        fig, ax = plt.subplots(figsize=(10, 4))
        # CRITICAL FIX 2: Corrected plotting variables to use 'df' (the uploaded data)
        ax.plot(df["time"], df["flux"], marker='.', linestyle='none', alpha=0.5, markersize=2)
        ax.set_xlabel("Time (Arbitrary Units)")
        ax.set_ylabel("Flux (Arbitrary Units)")
        ax.set_title("Uploaded Light Curve Analysis")
        st.pyplot(fig) 
        # CRITICAL FIX 1: Close the Matplotlib figure immediately to prevent memory leaks/crashes
        plt.close(fig)
        
        metrics = detect_transits(df)
        if metrics:
            st.success("Detected possible transit patterns!")
            ai_summary = generate_ai_summary("Uploaded File", metrics)
            st.info(ai_summary)
            if st.button("💾 Save Planet Profile"):
                save_planet_profile("Uploaded File", metrics, ai_summary)
        else:
            st.warning("No dips detected in uploaded data.")

# ===============================================================
# 6️⃣ MODE: SAVED PLANET PROFILES
# ===============================================================
elif mode == "Saved Planet Profiles":
    st.subheader("🗂️ Saved Planet Profiles")

    if st.session_state.planet_db:
        df = pd.DataFrame(st.session_state.planet_db)
        search = st.text_input("🔍 Search by TIC ID or keyword:")
        if search:
            df = df[df["TIC ID"].astype(str).str.contains(search, case=False)]
        st.dataframe(df)
    else:
        st.info("No planet profiles saved yet. Try analyzing a TIC ID first!")

# ===============================================================
# 7️⃣ MODE: AI PLANET ASSISTANT
# ===============================================================
elif mode == "AI Planet Assistant":
    st.subheader("🤖 Ask the AI Anything About Exoplanets")

    if not openai.api_key:
        st.warning("Please add your OpenAI API key in Streamlit Secrets to enable the AI assistant.")
    else:
        user_input = st.text_area("Ask a question about exoplanets or TESS data:")
        if st.button("💬 Ask"):
            if user_input.strip():
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": user_input}],
                        max_tokens=250
                    )
                    st.write(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please type a question first!")

# ===============================================================
# END OF APP
# ===============================================================
st.markdown("---")
st.caption("Built by Namann Alwaikar — Powered by NASA TESS, Astroquery, and OpenAI 🚀")
