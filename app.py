# app.py
# Streamlit app: Train a multi-output regressor (DO & Conductivity) and predict on uploaded CSVs

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import warnings
import plotly.express as px
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Water Quality DO & Conductivity", layout="wide")

# -------------------------- Constants --------------------------
TARGET_COLS_DEFAULT = ["dissolved_oxygen_(mg/l)", "conductivity_(µs/cm)"]
ID_COLS_DEFAULT = ["source_file", "sample_id"]

# -------------------------- Utilities --------------------------
def coerce_numeric(df: pd.DataFrame, skip_cols):
    out = df.copy()
    for c in out.columns:
        if c in skip_cols:
            continue
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(
                out[c].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.replace(r"[^\d\.\-eE]", "", regex=True),
                errors="coerce"
            )
    return out

def build_features_targets(df: pd.DataFrame, targets, id_cols):
    # Ensure targets exist
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    # Coerce numerics (excluding id & targets)
    df_num = coerce_numeric(df, skip_cols=set(id_cols + targets))

    # Build X, y
    X = df_num.drop(columns=[c for c in id_cols + targets if c in df_num.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    y = df[targets].copy()

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping IDs and targets.")

    # Clean NaNs/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median(numeric_only=True))

    return X, y

def get_base_regressor():
    try:
        import xgboost as xgb
        base_reg = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        used_model = "XGBRegressor"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        base_reg = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )
        used_model = "RandomForestRegressor (fallback)"
    return base_reg, used_model

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def average_feature_importance(model, feature_names):
    try:
        importances = np.zeros(len(feature_names))
        n = 0
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances += est.feature_importances_
                n += 1
        if n > 0:
            s = pd.Series(importances / n, index=feature_names).sort_values(ascending=False)
            return s
    except Exception:
        pass
    return None

def predict_on_dataframe(df_new, model_bundle):
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]
    targets = model_bundle["targets"]

    df_new = coerce_numeric(df_new, skip_cols=set([]))
    X_new = df_new.reindex(columns=feature_cols)
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(X_new.median(numeric_only=True))

    preds = model.predict(X_new)
    preds_df = pd.DataFrame(preds, columns=targets, index=df_new.index)
    return preds_df

# -------------------------- Sidebar --------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Upload CSVs for **training** and **prediction** in the tabs.")
test_size = st.sidebar.slider("Test size (hold-out split)", 0.1, 0.4, 0.2, 0.05)
run_loso = st.sidebar.checkbox("Leave-one-file-out validation (slower)", value=True)
target_cols = TARGET_COLS_DEFAULT  # Fixing per your requirement

# -------------------------- Tabs --------------------------
tab_train, tab_predict = st.tabs(["🎓 Train Model", "🔮 Predict with Model"])

# -------------------------- Train Tab --------------------------
with tab_train:
    st.header("🎓 Train on Uploaded CSV Files")
    train_files = st.file_uploader(
        "Upload one or more CSVs for training (they should include the target columns)",
        type=["csv"], accept_multiple_files=True, key="train_uploader"
    )

    if train_files:
        # Load and combine
        frames = []
        for f in train_files:
            try:
                df = pd.read_csv(f)
            except Exception:
                f.seek(0)
                df = pd.read_csv(f, encoding="latin-1")
            df.columns = [str(c).strip() for c in df.columns]
            df["source_file"] = f.name
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True, sort=False)
        st.success(f"Loaded {len(train_files)} files. Combined shape: {full_df.shape}")
        with st.expander("Preview combined data", expanded=False):
            st.dataframe(full_df.head(100))

        # Build features and targets
        try:
            X, y = build_features_targets(full_df, targets=target_cols, id_cols=ID_COLS_DEFAULT)
        except Exception as e:
            st.error(str(e))
            st.stop()

        feature_cols = list(X.columns)
        st.write("**Feature columns used:**", feature_cols)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model
        base_reg, used_model = get_base_regressor()
        model = MultiOutputRegressor(base_reg)
        model.fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        metrics_table = []
        for i, tgt in enumerate(target_cols):
            y_true_i = y_test.iloc[:, i].to_numpy()
            y_hat_i = np.asarray(y_pred)[:, i]
            metrics_table.append({
                "Target": tgt,
                "R²": round(r2_score(y_true_i, y_hat_i), 4),
                "RMSE": round(rmse(y_true_i, y_hat_i), 4)
            })
        st.subheader("📊 Hold-out Metrics")
        st.table(pd.DataFrame(metrics_table))
        st.caption(f"Base model used: **{used_model}**")

        # Optional leave-one-file-out validation
        if run_loso and full_df["source_file"].nunique() > 1:
            st.subheader("🧪 Leave-One-File-Out Validation")
            logo = LeaveOneGroupOut()
            groups = full_df["source_file"].values
            r2_scores = {t: [] for t in target_cols}
            rmse_scores = {t: [] for t in target_cols}

            for tr_idx, te_idx in logo.split(X, y, groups=groups):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                m = MultiOutputRegressor(get_base_regressor()[0])
                m.fit(X_tr, y_tr)
                y_hat = m.predict(X_te)
                for i, tgt in enumerate(target_cols):
                    r2_scores[tgt].append(r2_score(y_te.iloc[:, i], y_hat[:, i]))
                    rmse_scores[tgt].append(rmse(y_te.iloc[:, i], y_hat[:, i]))

            loso_df = pd.DataFrame({
                "Target": target_cols,
                "R² (mean)": [np.mean(r2_scores[t]) for t in target_cols],
                "RMSE (mean)": [np.mean(rmse_scores[t]) for t in target_cols],
                "Folds": [len(r2_scores[target_cols[0]])] * len(target_cols)
            })
            st.table(loso_df.round(4))

        # Save in session for Predict tab
        st.session_state["model_bundle"] = {
            "model": model,
            "feature_columns": feature_cols,
            "targets": target_cols
        }

        # ---------------- Visualizations ----------------
        st.subheader("📈 Visualizations")
        import plotly.express as px

        # Feature importance
        fi = average_feature_importance(model, feature_cols)
        if fi is not None:
            fig_fi = px.bar(fi.head(15)[::-1], orientation="h",
                            title="Top Feature Importances (avg across outputs)")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances not available for this estimator.")

        # Residual plots (per target)
        res_cols = st.columns(len(target_cols))
        for i, tgt in enumerate(target_cols):
            y_true_i = y_test.iloc[:, i].to_numpy()
            y_hat_i = np.asarray(y_pred)[:, i]
            residuals = y_true_i - y_hat_i
            df_res = pd.DataFrame({"Predicted": y_hat_i, "Residual": residuals})
            fig_res = px.scatter(df_res, x="Predicted", y="Residual", trendline="ols",
                                 title=f"Residuals vs Predicted — {tgt}")
            res_cols[i].plotly_chart(fig_res, use_container_width=True)

        # Pairwise relationships (sample to keep it light)
        sample = full_df.sample(min(1000, len(full_df)), random_state=42)
        plot_cols = [c for c in feature_cols[:3]] + target_cols  # limit to 3 features for clarity
        plot_cols = [c for c in plot_cols if c in sample.columns]
        if len(plot_cols) >= 3:
            fig_matrix = px.scatter_matrix(sample, dimensions=plot_cols,
                                           title="Scatter Matrix (sampled)")
            st.plotly_chart(fig_matrix, use_container_width=True)

        st.success("Training complete. Switch to the **Predict** tab to score new files.")

    else:
        st.info("Upload CSV files to start training.")

# -------------------------- Predict Tab --------------------------
with tab_predict:
    st.header("🔮 Predict with the Trained Model")
    if "model_bundle" not in st.session_state:
        st.warning("Train a model first in the 'Train Model' tab.")
        st.stop()

    model_bundle = st.session_state["model_bundle"]
    feature_cols = model_bundle["feature_columns"]
    targets = model_bundle["targets"]

    st.write("**Model expects these feature columns:**")
    st.code(", ".join(feature_cols) or "(none)")

    pred_files = st.file_uploader(
        "Upload CSV files to predict (columns will be aligned automatically)",
        type=["csv"], accept_multiple_files=True, key="pred_uploader"
    )

    if pred_files:
        all_outputs = []
        for f in pred_files:
            try:
                df_new = pd.read_csv(f)
            except Exception:
                f.seek(0)
                df_new = pd.read_csv(f, encoding="latin-1")
            df_new.columns = [str(c).strip() for c in df_new.columns]
            preds_df = predict_on_dataframe(df_new, model_bundle)
            out = df_new.copy()
            out[targets] = preds_df
            out["source_file"] = f.name
            all_outputs.append(out)

            st.subheader(f"Results: {f.name}")
            st.dataframe(out.head(100))

            # Download button for each file
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Predictions ({f.name})",
                data=csv_bytes,
                file_name=f"{f.name.rsplit('.',1)[0]}_with_predictions.csv",
                mime="text/csv",
            )

        # Combined output & visual summary
        combined_out = pd.concat(all_outputs, ignore_index=True)
        with st.expander("Preview combined predictions", expanded=False):
            st.dataframe(combined_out.head(200))

        st.subheader("📊 Prediction Visuals")
        import plotly.express as px

        # Distributions of predictions
        pred_cols = targets
        fig_hist = px.histogram(combined_out, x=pred_cols[0],
                                nbins=40, title=f"Distribution: {pred_cols[0]}")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_hist2 = px.histogram(combined_out, x=pred_cols[1],
                                 nbins=40, title=f"Distribution: {pred_cols[1]}")
        st.plotly_chart(fig_hist2, use_container_width=True)

        # Relationship between predicted DO & Conductivity
        fig_sc = px.scatter(combined_out, x=pred_cols[0], y=pred_cols[1],
                            color=combined_out.columns[0] if combined_out.columns.empty else None,
                            title=f"{pred_cols[0]} vs {pred_cols[1]} (Predicted)")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Optional: compare predictions vs a key feature if present (e.g., pH)
        if "ph" in combined_out.columns:
            fig_ph = px.scatter(combined_out, x="ph", y=pred_cols[1],
                                title="Conductivity (pred) vs pH",
                                trendline="ols")
            st.plotly_chart(fig_ph, use_container_width=True)

        # Combined download
        combined_csv = combined_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Combined Predictions",
            data=combined_csv,
            file_name="combined_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload CSV files to get predictions.")

# -------------------------- Footer --------------------------
st.caption("Tip: Keep column names consistent across files. Targets expected: "
           f"{', '.join(TARGET_COLS_DEFAULT)}")
