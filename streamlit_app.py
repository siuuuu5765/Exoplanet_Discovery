# streamlit_app.py
# Enhanced Exoplanet Discovery Hub — ISEF-ready features
# Author: Generated for user (enhancements: synthetic data, ML classifier, DB, reporting)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, os, uuid, json, base64
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import joblib
import sqlite3
from matplotlib.backends.backend_pdf import PdfPages

# Optional astro packages; guard use
try:
    from astropy.timeseries import BoxLeastSquares
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

# ----------------- Helpers / Utils -----------------
DB_PATH = "candidates.db"
MODEL_PATH = "rf_model.joblib"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id TEXT PRIMARY KEY,
        name TEXT,
        period REAL,
        depth REAL,
        radius_re REAL,
        confidence REAL,
        model_used TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_candidate(rec):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO candidates (id,name,period,depth,radius_re,confidence,model_used,created_at)
    VALUES (?,?,?,?,?,?,?,?)
    """, (rec['id'], rec['name'], rec['period'], rec['depth'], rec['radius_re'], rec['confidence'], rec.get('model_used',None), rec['created_at']))
    conn.commit()
    conn.close()

def load_candidates_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM candidates", conn)
    conn.close()
    return df

# ----------------- Synthetic dataset generation -----------------
def generate_synthetic_lightcurve(period=3.5, depth=0.01, duration=0.15, t0=1.0, baseline=1.0, noise=0.0008, days=27, cadence_min=30):
    """
    Create a synthetic lightcurve with transits (box-shaped) repeated over the baseline.
    Returns pandas.DataFrame with time, flux.
    """
    cadence_days = cadence_min / (60*24)
    t = np.arange(0, days, cadence_days)
    flux = np.ones_like(t) * baseline
    # add transits
    centers = np.arange(t0 - 5*period, t0 + 5*period, period)
    halfdur = duration/2.0
    for c in centers:
        intransit = np.abs(t - c) < halfdur
        flux[intransit] -= depth
    flux += np.random.normal(0, noise, size=len(t))
    return pd.DataFrame({"time": t, "flux": flux})

def generate_labelled_dataset(n_samples=500, classes=['planet','binary','noise'], seed=42):
    """
    Create a dataset of binned folded lightcurves and labels.
    For 'planet' -> small depth box; 'binary' -> deep V-shaped or double-dip; 'noise' -> pure noise.
    We'll return X (n_samples, n_bins) and y labels.
    """
    rng = np.random.RandomState(seed)
    n_bins = 100
    X = []
    y = []
    for i in range(n_samples):
        cls = rng.choice(classes, p=[0.5, 0.25, 0.25])
        if cls == 'planet':
            p = rng.uniform(0.5, 10)
            depth = rng.uniform(0.002, 0.02)
            dur = rng.uniform(0.05, 0.2)
            lc = generate_synthetic_lightcurve(period=p, depth=depth, duration=dur, noise=rng.uniform(0.0005,0.002))
            # fold and bin
            bins = bin_folded(lc, p, n_bins)
            X.append(bins)
            y.append(0)
        elif cls == 'binary':
            p = rng.uniform(0.2, 10)
            depth = rng.uniform(0.05, 0.3)
            dur = rng.uniform(0.02, 0.2)
            lc = generate_synthetic_lightcurve(period=p, depth=depth, duration=dur, noise=rng.uniform(0.001,0.005))
            # make binary shape slightly different: two dips or V-shape by smoothing
            bins = bin_folded(lc, p, n_bins)
            bins = np.minimum(bins, np.convolve(bins, np.ones(3)/3, mode='same'))  # sharpen
            X.append(bins)
            y.append(1)
        else:  # noise
            t = np.linspace(0, 27, int(27*48))
            flux = 1 + np.random.normal(0, 0.0015, size=len(t))
            lc = pd.DataFrame({"time": t, "flux": flux})
            # pick a random trial period for folding (harmless)
            p = rng.uniform(0.5, 10)
            bins = bin_folded(lc, p, n_bins)
            X.append(bins)
            y.append(2)
    return np.vstack(X), np.array(y)

def bin_folded(df, period, n_bins=100):
    """
    Phase-fold, then bin the folded curve into n_bins and return median flux per bin normalized to 1.
    """
    t = np.array(df['time'])
    f = np.array(df['flux'])
    phase = ((t % period) / period)
    bins = np.linspace(0,1,n_bins+1)
    inds = np.digitize(phase, bins) - 1
    binned = []
    for i in range(n_bins):
        sel = f[inds==i]
        if sel.size>0:
            binned.append(np.nanmedian(sel))
        else:
            binned.append(np.nan)
    arr = np.array(binned)
    # simple interpolation of nans
    nans = np.isnan(arr)
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
    # normalize
    arr = arr / np.median(arr)
    return arr

# ----------------- Feature / Model utilities -----------------
def train_model(X, y, n_estimators=200):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ----------------- Streamlit UI -----------------
st.set_page_config(layout="wide", page_title="Exoplanet Hub — ISEF Enhanced")
st.title("Exoplanet Discovery Hub — ISEF Enhancements")

init_db()
tab = st.tabs(["Analyze", "Train Model", "Experiments", "Database", "Export Report", "Chatbot (opt)"])

# ---------- ANALYZE TAB ----------
with tab[0]:
    st.header("Analyze a Lightcurve")
    uploaded = st.file_uploader("Upload CSV with 'time' and 'flux' (or use sample)", type=["csv"])
    use_sample = st.checkbox("Use sample synthetic lightcurve", value=True)
    if uploaded:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = generate_synthetic_lightcurve()
    else:
        st.info("Upload or enable sample to proceed.")
        df = None

    if df is not None:
        st.subheader("Lightcurve preview")
        st.dataframe(df.head())
        # detect with BLS if available
        if HAS_ASTROPY:
            try:
                t = np.array(df['time'])
                f = np.array(df['flux'])
                bls = BoxLeastSquares(t, f)
                periods = np.linspace(0.5, 10, 2000)
                blsres = bls.power(periods, 0.2)
                best_idx = np.argmax(blsres.power)
                best_period = blsres.period[best_idx]
                best_t0 = blsres.transit_time[best_idx]
                bls_snr = blsres.power[best_idx]
                st.success(f"BLS detection: period={best_period:.4f} d  (power={bls_snr:.3f})")
            except Exception as e:
                st.warning(f"BLS failed: {e}")
                best_period = float(st.number_input("Enter trial period (days)", value=3.5))
                best_t0 = 0.0
        else:
            best_period = float(st.number_input("Enter trial period (days)", value=3.5))
            best_t0 = 0.0

        # fold & plot
        phase = ((df['time'] - best_t0 + 0.5*best_period) % best_period) / best_period - 0.5
        plt.figure(figsize=(8,3))
        plt.scatter(phase, df['flux'], s=6)
        plt.xlabel("Phase")
        plt.ylabel("Flux")
        plt.title(f"Phase folded at P={best_period:.3f} d")
        st.pyplot(plt.gcf())
        plt.clf()

        # estimate depth and radius
        folded_df = pd.DataFrame({"phase": phase, "flux": df['flux']})
        # approximate depth as median outside - 5th percentile
        depth = abs(np.nanmedian(df['flux']) - np.nanpercentile(df['flux'], 5))
        st.metric("Estimated depth (fraction)", f"{depth:.6f}")
        Rs = st.number_input("Host star radius (R_sun)", value=1.0)
        RpRs = np.sqrt(depth) if depth>0 else 0.0
        RpRe = RpRs * Rs * 109.0
        st.metric("Estimated Planet Radius (Earth radii)", f"{RpRe:.2f}")

        # classify with model if exists
        model = load_model()
        if model is not None:
            # bin using best_period
            Xfeat = bin_folded(df, best_period, n_bins=100).reshape(1,-1)
            prob = model.predict_proba(Xfeat)[0]
            classes = model.classes_
            st.write("Model probabilities:")
            for c,p in zip(classes, prob):
                st.write(f"{c}: {p:.3f}")
            # choose predicted label
            pred = model.predict(Xfeat)[0]
            st.success(f"Model prediction: {pred}")
            confidence = float(prob.max())
        else:
            pred = "no-model"
            confidence = 0.0

        # Save candidate
        if st.button("Save Candidate"):
            rec = {
                "id": str(uuid.uuid4())[:8],
                "name": f"Candidate-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "period": float(best_period),
                "depth": float(depth),
                "radius_re": float(RpRe),
                "confidence": float(confidence),
                "model_used": os.path.basename(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
                "created_at": datetime.now().isoformat()
            }
            save_candidate(rec)
            st.success("Saved to local candidate DB.")

# ---------- TRAIN MODEL TAB ----------
with tab[1]:
    st.header("Train / Evaluate Classifier (Synthetic dataset)")
    st.write("Generate synthetic dataset and train a Random Forest classifier to distinguish planet / binary / noise.")
    n_samples = st.slider("Number of synthetic samples", 200, 5000, 1000, step=200)
    if st.button("Generate dataset & train"):
        X, y = generate_labelled_dataset(n_samples=n_samples, classes=['planet','binary','noise'])
        st.write("Dataset shape:", X.shape, y.shape)
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, MODEL_PATH)
        y_pred = clf.predict(X_test)
        st.write("Classification report:")
        st.text(classification_report(y_test, y_pred))
        st.write("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        # show feature importance (approx)
        importances = clf.feature_importances_
        plt.figure(figsize=(8,2))
        plt.plot(np.linspace(0,1,len(importances)), importances)
        plt.title("Feature importances over phase bins")
        st.pyplot(plt.gcf())
        plt.clf()
        st.success("Model trained and saved to disk (`rf_model.joblib`). Use in Analyze tab.")

# ---------- EXPERIMENTS TAB ----------
with tab[2]:
    st.header("Run Experiments (Detection limits)")
    st.write("Run controlled tests: vary depth / noise and measure detection/classification success.")
    n_trials = st.number_input("Trials per setting", value=50, min_value=5)
    depths = st.multiselect("Depths to test", [0.002,0.005,0.01,0.02,0.05], default=[0.002,0.005,0.01,0.02])
    noise_levels = st.multiselect("Noise (std) to test", [0.0005,0.001,0.002,0.005], default=[0.0005,0.001,0.002])
    if st.button("Run experiments") :
        model = load_model()
        if model is None:
            st.error("Train a model first in the Train Model tab.")
        else:
            rows = []
            for d in depths:
                for n in noise_levels:
                    successes = 0
                    for i in range(n_trials):
                        lc = generate_synthetic_lightcurve(depth=d, noise=n)
                        # attempt detection/classification
                        # choose a trial period 3.5 for consistency
                        feat = bin_folded(lc, 3.5, n_bins=100).reshape(1,-1)
                        pred = model.predict(feat)[0]
                        if pred == 0:  # planet label
                            successes += 1
                    acc = successes / n_trials
                    rows.append({"depth":d,"noise":n,"recall":acc})
            dfres = pd.DataFrame(rows)
            st.dataframe(dfres)
            st.line_chart(dfres.pivot(index="depth", columns="noise", values="recall"))
            # save experiment
            fn = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            dfres.to_csv(fn, index=False)
            st.success(f"Experiment finished and saved to {fn}")

# ---------- DATABASE TAB ----------
with tab[3]:
    st.header("Candidate Database")
    dfc = load_candidates_df()
    if dfc.empty:
        st.info("No candidates saved yet.")
    else:
        st.dataframe(dfc)
        csv = dfc.to_csv(index=False).encode('utf-8')
        st.download_button("Download DB CSV", csv, file_name="candidates_db.csv")

# ---------- EXPORT REPORT TAB ----------
with tab[4]:
    st.header("Export a PDF report for a candidate")
    dfc = load_candidates_df()
    if dfc.empty:
        st.info("No candidates to export.")
    else:
        id_sel = st.selectbox("Select candidate ID", dfc['id'])
        rec = dfc[dfc['id']==id_sel].iloc[0].to_dict()
        if st.button("Generate PDF summary"):
            pdf_name = f"candidate_{rec['id']}_summary.pdf"
            with PdfPages(pdf_name) as pdf:
                plt.figure(figsize=(8,4))
                plt.text(0.01, 0.9, f"Candidate: {rec['name']}", fontsize=14)
                plt.text(0.01, 0.85, f"Period: {rec['period']:.5f} d", fontsize=12)
                plt.text(0.01, 0.82, f"Depth: {rec['depth']:.6f}", fontsize=12)
                plt.text(0.01, 0.79, f"Radius (R⊕): {rec['radius_re']:.2f}", fontsize=12)
                plt.text(0.01, 0.76, f"Confidence: {rec['confidence']:.3f}", fontsize=12)
                plt.axis('off')
                pdf.savefig()
                plt.close()
            with open(pdf_name,'rb') as f:
                st.download_button("Download PDF", f.read(), file_name=pdf_name, mime="application/pdf")
            st.success("PDF generated.")



