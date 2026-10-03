import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
from xgboost import XGBRegressor
import hashlib
from datetime import datetime
from theme_manager import render_theme_selector

# Optional SHAP dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="DHAKA HEATMAP — Urban Heat Risk Monitoring System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SIDEBAR-FIRST AUTHENTICATION / ADMIN CONSOLE
# ============================================================

USERS = {
    "admin": {
        "name": "System Administrator",
        "email": "admin@dhakaheatmap.local",
        "role": "Admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "status": "Active",
    },
    "actor": {
        "name": "Heat Risk Actor",
        "email": "actor@dhakaheatmap.local",
        "role": "Actor",
        "password_hash": hashlib.sha256("actor123".encode()).hexdigest(),
        "status": "Active",
    },
    "analyst": {
        "name": "Heat Risk Analyst",
        "email": "analyst@dhakaheatmap.local",
        "role": "Analyst",
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "status": "Active",
    },
    "viewer": {
        "name": "Dashboard Viewer",
        "email": "viewer@dhakaheatmap.local",
        "role": "Viewer",
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "status": "Active",
    },
}

DEFAULT_SETTINGS = {
    "low_threshold": 33.0,
    "high_threshold": 66.0,
    "default_model": "Random Forest",
    "dashboard_refresh": True,
    "show_map": True,
    "show_ml_analysis": True,
}

def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "settings" not in st.session_state:
        st.session_state.settings = DEFAULT_SETTINGS.copy()

def log_activity(action, details=""):
    username = st.session_state.get("username")
    if username:
        user = USERS.get(username, {})
        st.session_state.activity_log.insert(
            0,
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User": username,
                "Role": user.get("role", ""),
                "Activity": action,
                "Details": details,
            },
        )
        st.session_state.activity_log = st.session_state.activity_log[:1000]

def authenticate(login_value, password):
    value = login_value.strip().lower()
    for username, user in USERS.items():
        if value in {username.lower(), user["email"].lower()}:
            if user["status"] != "Active":
                return None
            supplied = hashlib.sha256(password.encode()).hexdigest()
            return username if supplied == user["password_hash"] else False
    return False

def render_login():
    st.markdown(
        """
        <style>
        .login-page {
            min-height: 78vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            width: min(460px, 92vw);
            margin: 7vh auto 0 auto;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 22px;
            background: rgba(128,128,128,.07);
            box-shadow: 0 12px 40px rgba(0,0,0,.10);
        }
        .login-brand {
            width: min(460px, 92vw);
            box-sizing: border-box;
            text-align: center;
            margin: 7vh auto 1.5rem auto;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(128,128,128,.35);
            border-radius: 16px;
            background: rgba(128,128,128,.07);
            box-shadow: 0 8px 24px rgba(0,0,0,.08);
        }
        .login-brand h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .login-brand p {
            margin: .35rem 0 0 0;
            opacity: .75;
            font-size: 1rem;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown(
            """
            <div class="login-brand">
                <h1>🌡️ DHAKA HEATMAP</h1>
                <p>Urban Heat Risk Monitoring System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("full_page_login_form"):
            login_value = st.text_input(
                "Username / Email",
                placeholder="Enter username or email",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
            )

            submitted = st.form_submit_button(
                "🔐 Login",
                use_container_width=True,
            )

        if submitted:
            result = authenticate(login_value, password)

            if result:
                st.session_state.authenticated = True
                st.session_state.username = result
                log_activity("Login", "Successful login")
                st.rerun()
            elif result is None:
                st.error("Account is inactive. Contact an administrator.")
            else:
                st.error("❌ Wrong username/email or password.")

        st.caption("Authorized users only.")


def render_account_sidebar():
    user = USERS[st.session_state.username]
    with st.sidebar:
        st.markdown("## 👤 Account")
        st.write(f"**{user['name']}**")
        st.caption(f"{st.session_state.username} • {user['role']}")
        if st.button("Logout", use_container_width=True):
            log_activity("Logout", "User logged out")
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

def render_admin_dashboard():
    st.header("🛠️ Admin Dashboard")
    total_actors = sum(u["role"] == "Actor" for u in USERS.values())
    active_users = sum(u["status"] == "Active" for u in USERS.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Actors", total_actors)
    c2.metric("Active Users", active_users)
    c3.metric("Total Accounts", len(USERS))

def render_actor_management():
    st.header("👥 User Management")
    rows = [
        {
            "Username": name,
            "Email": user["email"],
            "Name": user["name"],
            "Role": user["role"],
            "Account Status": user["status"],
        }
        for name, user in USERS.items()
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    candidates = [name for name in USERS if name != "admin" and USERS[name]["status"] == "Active"]
    if candidates:
        selected = st.selectbox("Select user to remove", candidates)
        if st.button("Remove User", type="secondary"):
            USERS[selected]["status"] = "Inactive"
            log_activity("Actor Management", f"Removed user: {selected}")
            st.success(f"{selected} has been deactivated.")
            st.rerun()
    else:
        st.info("No active removable users.")

def render_activity_log():
    st.header("📋 Activity Log")
    st.caption(
        "Login • Logout • Heatmap Viewed • Area Searched • "
        "ML Prediction Request • Report Generated"
    )
    logs = st.session_state.activity_log
    if logs:
        st.dataframe(logs, use_container_width=True, hide_index=True)
        csv = pd.DataFrame(logs).to_csv(index=False)
        st.download_button(
            "⬇️ Generate/Export Activity Report",
            data=csv,
            file_name="dhaka_heatmap_activity_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No activity recorded yet.")

def render_system_settings():
    st.header("⚙️ System Settings")
    settings = st.session_state.settings

    c1, c2 = st.columns(2)
    with c1:
        low = st.number_input("Low threshold", 0.0, 100.0, float(settings["low_threshold"]))
    with c2:
        high = st.number_input("High threshold", 0.0, 100.0, float(settings["high_threshold"]))

    available = [n for n in ["Linear Regression", "Random Forest", "XGBoost"] if n in models]
    if not available:
        available = ["Random Forest"]
    current = settings["default_model"] if settings["default_model"] in available else available[0]
    default_model = st.selectbox("Default model", available, index=available.index(current))

    refresh = st.checkbox("Dashboard refresh controls", value=settings["dashboard_refresh"])
    show_map = st.checkbox("Show heatmap", value=settings["show_map"])
    show_ml = st.checkbox("Show ML Analysis", value=settings["show_ml_analysis"])

    if st.button("Save System Settings", type="primary"):
        if low >= high:
            st.error("Low threshold must be lower than High threshold.")
        else:
            settings.update({
                "low_threshold": low,
                "high_threshold": high,
                "default_model": default_model,
                "dashboard_refresh": refresh,
                "show_map": show_map,
                "show_ml_analysis": show_ml,
            })
            log_activity(
                "System Settings",
                f"Thresholds={low}/{high}; Default model={default_model}",
            )
            st.success("System settings saved.")

def render_profile():
    user = USERS[st.session_state.username]
    st.header("👤 Profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Username", st.session_state.username)
    c2.metric("Role", user["role"])
    c3.metric("Status", user["status"])
    st.write(f"**Name:** {user['name']}")
    st.write(f"**Email:** {user['email']}")

init_auth_state()

if not st.session_state.authenticated:
    render_login()
    st.stop()


# ============================================================
# DATA / MODEL PATHS AND LOADERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RISK_FILE = BASE_DIR / "dhaka_heat_risk_data.csv"
FEATURE_FILE = BASE_DIR / "Dhaka_Heatmap_Grid_Features.csv"
FINAL_GEOJSON = BASE_DIR / "Dhaka_Heatmap_Grid_Final.geojson"

RF_FILE = BASE_DIR / "random_forest_model.joblib"
LR_FILE = BASE_DIR / "linear_regression_model.joblib"
XGB_FILE = BASE_DIR / "xgboost_model.json"

FEATURES = ["NDVI", "NDBI", "NDWI"]


# Metrics reported by the original Colab experiment.
NOTEBOOK_RESULTS = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            "MAE": 0.7457,
            "RMSE": 0.9686,
            "R²": 0.7508,
        },
        {
            "Model": "Random Forest",
            "MAE": 0.7654,
            "RMSE": 0.9803,
            "R²": 0.7448,
        },
        {
            "Model": "XGBoost",
            "MAE": 0.7673,
            "RMSE": 0.9965,
            "R²": 0.7363,
        },
    ]
)


@st.cache_data
def calculate_single_risk(lst, ndvi, ndbi, ndwi):
    """Calculate heat-risk score using the notebook weights.

    The source CSV may contain either observed LST or only predicted_LST.
    For interactive predictions, use predicted_LST as the LST reference
    when observed LST is unavailable.
    """
    if "LST" in df.columns:
        lst_series = pd.to_numeric(df["LST"], errors="coerce").dropna()
    elif "predicted_LST" in df.columns:
        lst_series = pd.to_numeric(df["predicted_LST"], errors="coerce").dropna()
    else:
        # Last-resort fallback for interactive prediction if neither LST field
        # exists in the tabular source.
        lst_series = pd.Series([float(lst)])

    def get_range(column, fallback_value):
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.min()), float(series.max())
        value = float(fallback_value)
        return value - 1.0, value + 1.0

    lst_min, lst_max = float(lst_series.min()), float(lst_series.max())
    ndvi_min, ndvi_max = get_range("NDVI", ndvi)
    ndbi_min, ndbi_max = get_range("NDBI", ndbi)
    ndwi_min, ndwi_max = get_range("NDWI", ndwi)

    def normalize(value, low, high):
        if high == low:
            return 50.0
        return float(np.clip((float(value) - low) / (high - low) * 100.0, 0, 100))

    # Heat-risk direction:
    # higher LST -> higher risk
    # lower NDVI -> higher risk
    # higher NDBI -> higher risk
    # lower NDWI -> higher risk
    lst_score = normalize(lst, lst_min, lst_max)
    ndvi_score = 100.0 - normalize(ndvi, ndvi_min, ndvi_max)
    ndbi_score = normalize(ndbi, ndbi_min, ndbi_max)
    ndwi_score = 100.0 - normalize(ndwi, ndwi_min, ndwi_max)

    score = (
        0.50 * lst_score
        + 0.15 * ndvi_score
        + 0.25 * ndbi_score
        + 0.10 * ndwi_score
    )
    score = float(np.clip(score, 0, 100))

    if score < 33:
        category = "Low"
    elif score < 66:
        category = "Medium"
    else:
        category = "High"

    return score, category


def load_risk_data():
    df = pd.read_csv(RISK_FILE)
    df.columns = df.columns.str.strip()

    if "risk_category" not in df.columns:
        if "risk_catagory" in df.columns:
            df["risk_category"] = df["risk_catagory"]
        elif "heat_risk_score" in df.columns:
            df["risk_category"] = pd.cut(
                df["heat_risk_score"],
                bins=[-np.inf, 33, 66, np.inf],
                labels=["Low", "Medium", "High"],
            )

    return df


@st.cache_data
def load_feature_data():
    df = pd.read_csv(FEATURE_FILE)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_geojson():
    with open(FINAL_GEOJSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    models = {}
    model_errors = {}

    # Scikit-learn models
    for model_name, model_file in {
        "Linear Regression": LR_FILE,
        "Random Forest": RF_FILE,
    }.items():
        try:
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_file.name}"
                )
            models[model_name] = joblib.load(model_file)
        except Exception as exc:
            model_errors[model_name] = f"{type(exc).__name__}: {exc}"

    # XGBoost native JSON model
    try:
        if not XGB_FILE.exists():
            raise FileNotFoundError(
                f"Model file not found: {XGB_FILE.name}"
            )
        xgb_model = XGBRegressor()
        xgb_model.load_model(XGB_FILE)
        models["XGBoost"] = xgb_model
    except Exception as exc:
        model_errors["XGBoost"] = f"{type(exc).__name__}: {exc}"

    return models, model_errors

df = load_risk_data()
feature_df = load_feature_data()
geojson_data = load_geojson()
models, model_errors = load_models()

# Build a dashboard-level grid ID from the authoritative feature CSV.
# The risk CSV has no grid identifier, so do NOT look for grid_id in df.
if "system:index" not in feature_df.columns:
    raise ValueError(
        "Grid ID source column 'system:index' was not found in "
        "Dhaka_Heatmap_Grid_Features.csv."
    )

if len(df) != len(feature_df):
    raise ValueError(
        f"Risk data rows ({len(df)}) do not match feature rows "
        f"({len(feature_df)}), so grid IDs cannot be safely aligned."
    )

dashboard_df = df.copy()
dashboard_df["grid_id"] = feature_df["system:index"].astype(str).values

if model_errors:
    for model_name, error_message in model_errors.items():
        st.warning(
            f"{model_name} model could not be loaded. "
            f"The dashboard will continue without this model. "
            f"Details: {error_message}"
        )



# ============================================================
# ROLE-SPECIFIC SCREENS
# ============================================================

def render_dashboard_statistics():
    st.header("📊 Dashboard Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid Cells", f"{len(df):,}")
    c2.metric("Average LST", f"{df['predicted_LST'].mean():.2f} °C")
    c3.metric("Average Heat Risk", f"{df['heat_risk_score'].mean():.2f}")
    high_risk = int((df["risk_category"].astype(str) == "High").sum())
    c4.metric("High Risk Cells", f"{high_risk:,}")

    st.subheader("Risk-zone summary")
    counts = (
        df["risk_category"].astype(str).value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Grid Cells")
    )
    fig = px.bar(
        counts,
        x="Risk Category",
        y="Grid Cells",
        title="Dhaka Heat Risk Zones",
        category_orders={"Risk Category": ["Low", "Medium", "High"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_environmental_analysis():
    st.header("🌿 Environmental Data Analysis")
    st.write(
        "Inspect the four core variables used in the heat-risk workflow: "
        "LST, NDVI, NDBI and NDWI."
    )

    available_cols = [c for c in ["LST", "NDVI", "NDBI", "NDWI"] if c in feature_df.columns]
    st.dataframe(
        feature_df[available_cols].describe().T.round(4),
        use_container_width=True,
    )

    feature = st.selectbox("Select environmental variable", ["NDVI", "NDBI", "NDWI"], key="env_feature_role")
    if feature in feature_df.columns and "LST" in feature_df.columns:
        plot_df = feature_df[[feature, "LST"]].dropna()
        fig = px.scatter(
            plot_df,
            x=feature,
            y="LST",
            opacity=0.6,
            title=f"{feature} vs Observed LST",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_area_grid_details():
    st.header("📍 Area / Grid Details")
    st.write(
        "Search a grid identifier and inspect its heat-risk and environmental "
        "attributes."
    )

    grid_ids = (
        dashboard_df["grid_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not grid_ids:
        st.warning("No grid IDs are available.")
        return

    selected = st.selectbox(
        "Select Grid ID",
        grid_ids,
        key="area_grid_id",
    )

    risk_row = dashboard_df[
        dashboard_df["grid_id"].astype(str) == str(selected)
    ].copy()

    feature_row = feature_df[
        feature_df["system:index"].astype(str) == str(selected)
    ].copy()

    if risk_row.empty or feature_row.empty:
        st.error(f"Grid ID '{selected}' could not be matched to the datasets.")
        return

    # Merge the matching row from both datasets so the user gets the
    # complete set of grid/risk/environment attributes.
    detail = risk_row.iloc[0].to_dict()
    feature_values = feature_row.iloc[0].to_dict()

    for col in ["LST", "NDVI", "NDBI", "NDWI"]:
        if col in feature_values:
            detail[col] = feature_values[col]

    detail["grid_id"] = str(selected)

    ordered = [
        "grid_id",
        "LST",
        "predicted_LST",
        "NDVI",
        "NDBI",
        "NDWI",
        "heat_risk_score",
        "risk_category",
    ]
    detail_df = pd.DataFrame(
        [{col: detail.get(col) for col in ordered}]
    )

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True,
    )
    log_activity("Area Searched", f"Grid ID: {selected}")

def render_reports_visualization():
    st.header("📊 Reports / Visualization")

    risk_counts = (
        df["risk_category"].astype(str)
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Grid Cells")
    )
    fig_risk = px.bar(
        risk_counts,
        x="Risk Category",
        y="Grid Cells",
        title="Heat Risk Zone Distribution",
        category_orders={"Risk Category": ["Low", "Medium", "High"]},
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    model_table = NOTEBOOK_RESULTS.copy()
    st.subheader("Model Performance")
    st.dataframe(
        model_table.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    report_csv = df.to_csv(index=False).encode("utf-8")
    if st.download_button(
        "⬇️ Export Heat Risk Report",
        report_csv,
        file_name="dhaka_heat_risk_report.csv",
        mime="text/csv",
        use_container_width=True,
    ):
        log_activity("Report Generated", "Heat risk report exported")


def render_system_model_status():
    st.header("🖥️ System & Model Status")

    status_cols = st.columns(4)
    status_cols[0].metric("Risk Data Rows", f"{len(df):,}")
    status_cols[1].metric("Feature Rows", f"{len(feature_df):,}")
    status_cols[2].metric("GeoJSON Grids", f"{len(geojson_data.get('features', [])):,}")
    status_cols[3].metric("Loaded Models", f"{len(models):,}")

    st.subheader("Model availability")
    model_rows = []
    for model_name in ["Linear Regression", "Random Forest", "XGBoost"]:
        if model_name in models:
            model_rows.append({"Model": model_name, "Status": "Loaded"})
        else:
            model_rows.append({
                "Model": model_name,
                "Status": f"Error: {model_errors.get(model_name, 'Unavailable')}",
            })
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    st.subheader("Current system configuration")
    st.json(st.session_state.settings)


def render_ml_prediction():
    st.header("🤖 Interactive ML Prediction")
    st.write(
        "Enter NDVI, NDBI and NDWI values to obtain an LST prediction. "
        "The configured default model is preselected."
    )

    available_models = list(models.keys())
    if not available_models:
        st.error("No ML model is currently available.")
        return

    configured = st.session_state.settings.get("default_model", "Random Forest")
    default_model = configured if configured in available_models else available_models[0]

    model_name = st.selectbox(
        "Prediction model",
        available_models,
        index=available_models.index(default_model),
        key="actor_prediction_model",
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        ndvi_value = st.number_input(
            "NDVI",
            min_value=float(feature_df["NDVI"].min()),
            max_value=float(feature_df["NDVI"].max()),
            value=float(feature_df["NDVI"].median()),
            format="%.5f",
            key="actor_ndvi",
        )
    with p2:
        ndbi_value = st.number_input(
            "NDBI",
            min_value=float(feature_df["NDBI"].min()),
            max_value=float(feature_df["NDBI"].max()),
            value=float(feature_df["NDBI"].median()),
            format="%.5f",
            key="actor_ndbi",
        )
    with p3:
        ndwi_value = st.number_input(
            "NDWI",
            min_value=float(feature_df["NDWI"].min()),
            max_value=float(feature_df["NDWI"].max()),
            value=float(feature_df["NDWI"].median()),
            format="%.5f",
            key="actor_ndwi",
        )

    if st.button("Run ML Prediction", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [[ndvi_value, ndbi_value, ndwi_value]],
            columns=FEATURES,
        )
        predicted_lst = float(models[model_name].predict(input_df)[0])
        score, category = calculate_single_risk(
            predicted_lst, ndvi_value, ndbi_value, ndwi_value
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted LST", f"{predicted_lst:.2f} °C")
        r2.metric("Heat Risk Score", f"{score:.2f}")
        r3.metric("Risk Category", category)

        st.progress(int(round(score)))
        st.session_state["last_prediction"] = predicted_lst
        log_activity(
            "ML Prediction Request",
            f"{model_name}: predicted LST={predicted_lst:.3f}, risk={category}",
        )


def render_heat_risk_zones():
    st.header("🔥 Heat Risk Zones")
    counts = (
        df["risk_category"].astype(str)
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Grid Cells")
    )
    st.dataframe(counts, use_container_width=True, hide_index=True)

    fig = px.bar(
        counts,
        x="Risk Category",
        y="Grid Cells",
        title="Heat Risk Zone Distribution",
        category_orders={"Risk Category": ["Low", "Medium", "High"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_login_account():
    render_profile()





def render_heat_risk_map():
    # Filter controls stay inside the drawer so this screen is fully
    # controlled from the role-based sidebar.
    risk_options = ["Low", "Medium", "High"]
    available_risks = [
        x for x in risk_options
        if x in df["risk_category"].astype(str).unique()
    ]

    with st.sidebar:
        st.markdown("### 🎚️ Map Filters")
        risk_filter = st.multiselect(
            "Risk Category",
            options=available_risks,
            default=available_risks,
            key="role_map_risk_filter",
        )

    filtered_df = df[
        df["risk_category"].astype(str).isin(risk_filter)
    ].copy()

    st.subheader("📊 Heat Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Grid Cells", f"{len(df):,}")

    with col2:
        st.metric("Average LST", f"{df['predicted_LST'].mean():.2f} °C")

    with col3:
        st.metric("Average Heat Risk", f"{df['heat_risk_score'].mean():.2f}")

    with col4:
        high_risk = int((df["risk_category"].astype(str) == "High").sum())
        st.metric("High Risk Cells", f"{high_risk:,}")

    st.subheader("🗺️ Interactive Dhaka Heat Risk Map")
    st.write(
        "Each grid represents approximately a 500m × 500m area. "
        "Click a grid cell to inspect its environmental and heat-risk information."
    )

    # Work on a copy so the cached GeoJSON is never modified permanently.
    map_geojson = json.loads(json.dumps(geojson_data))

    # Build fast lookup by grid_id.
    if "system:index" in feature_df.columns:
        feature_lookup = feature_df.set_index(
            feature_df["system:index"].astype(str)
        ).to_dict("index")
    else:
        feature_lookup = {}

    risk_lookup = df.copy()
    if "system:index" in feature_df.columns:
        risk_lookup["system:index"] = feature_df["NDVI"].astype(str) if False else feature_df["system:index"].astype(str).values
        risk_lookup = risk_lookup.set_index("system:index").to_dict("index")
    else:
        risk_lookup = {}

    for feature in map_geojson["features"]:
        props = feature.setdefault("properties", {})
        grid_id = str(props.get("grid_id", ""))

        if grid_id in feature_lookup:
            row = feature_lookup[grid_id]
            for col in ["LST", "NDVI", "NDBI", "NDWI"]:
                if col in row and pd.notna(row[col]):
                    props[col] = float(row[col])

        if grid_id in risk_lookup:
            row = risk_lookup[grid_id]
            for col in ["predicted_LST", "heat_risk_score"]:
                if col in row and pd.notna(row[col]):
                    props[col] = float(row[col])
            if "risk_category" in row:
                props["risk_category"] = str(row["risk_category"])

    def get_color(risk):
        return {
            "High": "#d73027",
            "Medium": "#fee08b",
            "Low": "#1a9850",
        }.get(str(risk), "#cccccc")

    m = folium.Map(
        location=[23.78, 90.40],
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    def style_function(feature):
        risk = feature["properties"].get("risk_category", "Unknown")
        return {
            "fillColor": get_color(risk),
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.55,
        }

    def highlight_function(feature):
        return {
            "weight": 2,
            "color": "#000000",
            "fillOpacity": 0.8,
        }

    folium.GeoJson(
        map_geojson,
        name="500m Heat Risk Grid",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "grid_id",
                "predicted_LST",
                "heat_risk_score",
                "risk_category",
            ],
            aliases=[
                "Grid ID:",
                "LST (°C):",
                "Heat Risk Score:",
                "Risk Category:",
            ],
            localize=True,
            sticky=False,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "grid_id",
                "LST",
                "predicted_LST",
                "NDVI",
                "NDBI",
                "NDWI",
                "heat_risk_score",
                "risk_category",
            ],
            aliases=[
                "Grid ID:",
                "Observed LST (°C):",
                "Predicted LST (°C):",
                "NDVI:",
                "NDBI:",
                "NDWI:",
                "Heat Risk Score:",
                "Risk Category:",
            ],
            localize=True,
            labels=True,
        ),
    ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 180px;
        background-color: white;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 14px;
        padding: 10px;
    ">
    <b>Heat Risk</b><br><br>
    <span style="color:#1a9850;">■</span> Low<br>
    <span style="color:#fee08b;">■</span> Medium<br>
    <span style="color:#d73027;">■</span> High
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=650, returned_objects=[])

    st.subheader("🔥 Heat Risk Distribution")

    risk_counts = (
        filtered_df["risk_category"]
        .astype(str)
        .value_counts()
        .reindex(risk_options, fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk Category", "Grid Cells"]

    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Grid Cells",
        title="Dhaka Heat Risk Categories",
        category_orders={"Risk Category": risk_options},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Environmental Features")

    feature = st.selectbox(
        "Select Feature",
        FEATURES,
        key="map_feature",
    )

    fig2 = px.scatter(
        filtered_df,
        x=feature,
        y="predicted_LST",
        color="risk_category",
        title=f"{feature} vs Predicted LST",
        opacity=0.6,
        hover_data=["heat_risk_score"],
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_ml_analysis():
    st.header("🤖 Machine Learning Analysis")

    st.write(
        "The notebook uses NDVI, NDBI and NDWI as predictors of LST, "
        "with an 80/20 train-test split and random_state=42."
    )

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.subheader("1. Model Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("ML Samples", f"{len(feature_df):,}")

    with c2:
        st.metric("Training Samples", f"{int(len(feature_df) * 0.8):,}")

    with c3:
        st.metric("Testing Samples", f"{len(feature_df) - int(len(feature_df) * 0.8):,}")

    with c4:
        st.metric("Predictor Variables", "3")

    st.markdown(
        """
        **Predictors:** NDVI, NDBI, NDWI  
        **Target:** LST  
        **Models:** Linear Regression, Random Forest Regressor, XGBoost Regressor  
        **Split:** 80% training / 20% testing  
        **Random state:** 42
        """
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("2. Model Comparison")

    st.markdown("**Metrics reported by the original notebook**")

    st.dataframe(
        NOTEBOOK_RESULTS.style.format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    metric = st.selectbox(
        "Metric to visualize",
        ["MAE", "RMSE", "R²"],
        key="notebook_metric",
    )

    fig_metrics = px.bar(
        NOTEBOOK_RESULTS,
        x="Model",
        y=metric,
        text_auto=".4f",
        title=f"Notebook-reported {metric}",
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # --------------------------------------------------------
    # RECOMPUTED METRICS FROM SUPPLIED MODEL FILES
    # --------------------------------------------------------

    st.subheader("3. Validation of Supplied Model Files")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = feature_df[FEATURES].copy()
    y = feature_df["LST"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    supplied_rows = []

    for model_name, model in models.items():
        pred = model.predict(X_test)
        supplied_rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R²": r2_score(y_test, pred),
            }
        )

    supplied_results = pd.DataFrame(supplied_rows)

    st.dataframe(
        supplied_results.style.format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These values are recomputed from the currently supplied .joblib files "
        "against the original LST field using the notebook's 80/20 split. "
        "They may differ from the notebook's printed values if the saved model "
        "was produced by a later/tuned experiment."
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader("4. Prediction Performance")

    selected_model_name = st.selectbox(
        "Select model",
        list(models.keys()),
        index=1,
        key="performance_model",
    )

    selected_model = models[selected_model_name]
    test_pred = selected_model.predict(X_test)

    perf_df = pd.DataFrame(
        {
            "Observed LST": y_test.values,
            "Predicted LST": test_pred,
        }
    )

    fig_perf = px.scatter(
        perf_df,
        x="Observed LST",
        y="Predicted LST",
        title=f"Observed vs Predicted LST — {selected_model_name}",
        opacity=0.65,
    )

    min_val = min(perf_df.min())
    max_val = max(perf_df.max())

    fig_perf.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect prediction",
        )
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("MAE", f"{mean_absolute_error(y_test, test_pred):.4f} °C")
    with metric_cols[1]:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, test_pred)):.4f} °C")
    with metric_cols[2]:
        st.metric("R²", f"{r2_score(y_test, test_pred):.4f}")

    errors = y_test.values - test_pred

    fig_error = px.histogram(
        x=errors,
        nbins=35,
        title=f"Prediction Error Distribution — {selected_model_name}",
        labels={"x": "Observed − Predicted LST (°C)", "y": "Count"},
    )
    st.plotly_chart(fig_error, use_container_width=True)

    # --------------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------------

    st.subheader("5. Feature Analysis")

    feature_choice = st.selectbox(
        "Feature",
        FEATURES,
        key="ml_feature",
    )

    feature_plot_df = feature_df[[feature_choice, "LST"]].dropna()

    fig_feature = px.scatter(
        feature_plot_df,
        x=feature_choice,
        y="LST",
        opacity=0.55,
        title=f"{feature_choice} vs Observed LST",
    )
    st.plotly_chart(fig_feature, use_container_width=True)

    rf_model = models["Random Forest"]

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    fig_importance = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        text_auto=".3f",
        title="Random Forest Feature Importance",
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader("6. SHAP Explainability")

    if not SHAP_AVAILABLE:
        st.warning(
            "SHAP is not installed. Add `shap` to requirements.txt and restart "
            "the Streamlit app to enable SHAP explanations."
        )
    else:
        shap_sample_size = st.slider(
            "SHAP sample size",
            min_value=100,
            max_value=min(1000, len(X_test)),
            value=min(500, len(X_test)),
            step=50,
            key="shap_sample",
        )

        shap_X = X_test.sample(
            n=shap_sample_size,
            random_state=42,
        )

        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(shap_X)

        shap_abs = np.abs(shap_values).mean(axis=0)

        shap_bar_df = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Mean |SHAP value|": shap_abs,
            }
        ).sort_values("Mean |SHAP value|", ascending=False)

        fig_shap_bar = px.bar(
            shap_bar_df,
            x="Feature",
            y="Mean |SHAP value|",
            title="SHAP Global Feature Importance",
            text_auto=".4f",
        )
        st.plotly_chart(fig_shap_bar, use_container_width=True)

        st.caption(
            "Positive SHAP values push a prediction higher than the model's "
            "baseline; negative values push it lower."
        )

        # Plotly-compatible SHAP beeswarm-style data.
        long_rows = []
        for j, feature_name in enumerate(FEATURES):
            for i in range(len(shap_X)):
                long_rows.append(
                    {
                        "Feature": feature_name,
                        "SHAP value": shap_values[i, j],
                        "Feature value": shap_X.iloc[i, j],
                    }
                )

        shap_long = pd.DataFrame(long_rows)

        fig_shap = px.scatter(
            shap_long,
            x="SHAP value",
            y="Feature",
            color="Feature value",
            title="SHAP Feature Effects",
            opacity=0.55,
            hover_data=["Feature value"],
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # --------------------------------------------------------
    # INTERACTIVE PREDICTION
    # --------------------------------------------------------

    st.subheader("7. Interactive LST Prediction")

    st.write(
        "Enter NDVI, NDBI and NDWI values to obtain an LST prediction from "
        "the supplied Random Forest model."
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        ndvi_value = st.number_input(
            "NDVI",
            min_value=float(feature_df["NDVI"].min()),
            max_value=float(feature_df["NDVI"].max()),
            value=float(feature_df["NDVI"].median()),
            format="%.5f",
        )

    with p2:
        ndbi_value = st.number_input(
            "NDBI",
            min_value=float(feature_df["NDBI"].min()),
            max_value=float(feature_df["NDBI"].max()),
            value=float(feature_df["NDBI"].median()),
            format="%.5f",
        )

    with p3:
        ndwi_value = st.number_input(
            "NDWI",
            min_value=float(feature_df["NDWI"].min()),
            max_value=float(feature_df["NDWI"].max()),
            value=float(feature_df["NDWI"].median()),
            format="%.5f",
        )

    input_df = pd.DataFrame(
        [[ndvi_value, ndbi_value, ndwi_value]],
        columns=FEATURES,
    )

    predicted_lst = float(rf_model.predict(input_df)[0])

    st.session_state["last_prediction"] = predicted_lst

    log_activity("ML Prediction Request", f"Random Forest prediction: {predicted_lst:.3f}")

    interactive_score, interactive_category = calculate_single_risk(
        predicted_lst,
        ndvi_value,
        ndbi_value,
        ndwi_value,
    )

    result_cols = st.columns(3)

    with result_cols[0]:
        st.metric("Predicted LST", f"{predicted_lst:.2f} °C")

    with result_cols[1]:
        st.metric("Heat Risk Score", f"{interactive_score:.2f}")

    with result_cols[2]:
        st.metric("Risk Category", interactive_category)

    st.progress(int(round(interactive_score)))

    st.info(
        "The interactive heat-risk score uses the same min-max normalization "
        "and 50/15/25/10 weighting scheme as the notebook. For a single new "
        "input, the normalization ranges come from the supplied dashboard dataset."
    )


def render_data_explorer():
    risk_options = ["Low", "Medium", "High"]
    available_risks = [
        x for x in risk_options
        if x in df["risk_category"].astype(str).unique()
    ]
    filtered_df = dashboard_df[
        dashboard_df["risk_category"].astype(str).isin(available_risks)
    ].copy()

    st.header("📋 Data Explorer")

    st.write(
        "The feature dataset contains the original observed LST together with "
        "the three ML predictors. The risk dataset contains Random Forest "
        "predictions and the derived heat-risk score."
    )

    view = st.radio(
        "Dataset",
        ["Heat-risk data", "Grid feature data"],
        horizontal=True,
    )

    if view == "Heat-risk data":
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_download = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download filtered heat-risk data",
            csv_download,
            file_name="dhaka_heat_risk_filtered.csv",
            mime="text/csv",
        )

    else:
        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_download = feature_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download grid feature data",
            csv_download,
            file_name="Dhaka_Heatmap_Grid_Features.csv",
            mime="text/csv",
        )

    st.subheader("Dataset Summary")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric("Feature rows", f"{len(feature_df):,}")

    with summary_cols[1]:
        st.metric("Risk rows", f"{len(df):,}")

    with summary_cols[2]:
        st.metric("GeoJSON grids", f"{len(geojson_data.get('features', [])):,}")

    with summary_cols[3]:
        st.metric("Missing feature values", int(feature_df[FEATURES].isna().sum().sum()))



# ============================================================
# SLIDE-OUT SIDEBAR DRAWER + ROLE-BASED NAVIGATION
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.22);
        box-shadow: 8px 0 28px rgba(0,0,0,.18);
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stRadio > div > label {
        border-radius: 10px;
    }
    .role-badge {
        display: inline-block;
        padding: .35rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.35);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .03em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

current_user = USERS[st.session_state.username]
role = current_user["role"]

ROLE_MENUS = {
    "Admin": [
        ("🏠", "Dashboard"),
        ("👥", "User Management"),
        ("⚙️", "Settings"),
        ("📝", "Login / Activity Monitoring"),
        ("🖥️", "System & Model Status"),
    ],
    "Actor": [
        ("🗺️", "Interactive Map Check"),
        ("🔥", "Heat Risk Analysis"),
        ("🌿", "LST / NDVI / NDBI / NDWI Check"),
        ("🤖", "ML Prediction"),
        ("📋", "Activity Log"),
    ],
    "Analyst": [
        ("🗺️", "Heatmap Analysis"),
        ("🔎", "Area / Grid Details"),
        ("🤖", "ML Model Results"),
        ("🌿", "Environmental Data Analysis"),
        ("📊", "Reports / Visualization"),
    ],
    "Viewer": [
        ("🔐", "Login / Account"),
        ("🗺️", "Interactive Map View"),
        ("🔥", "Heat Risk Zones"),
        ("📍", "Basic Area Information"),
        ("📊", "Dashboard Statistics"),
    ],
}

with st.sidebar:
    st.markdown("## ☰ DHAKA HEATMAP")
    st.markdown(f'<div class="role-badge">{role} USER</div>', unsafe_allow_html=True)
    st.caption(f"{current_user['name']} • {st.session_state.username}")
    st.divider()

    menu_items = ROLE_MENUS[role]
    labels = [f"{icon}  {label}" for icon, label in menu_items]
    menu_key = f"role_navigation_{role}"
    selected_label = st.radio(
        "Navigation",
        labels,
        key=menu_key,
    )
    selected_page = selected_label.split("  ", 1)[1]

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        log_activity("Logout", "User logged out")
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# Sidebar pages are role-limited by construction.
if role == "Admin":
    if selected_page == "Dashboard":
        render_dashboard_statistics()
    elif selected_page == "User Management":
        render_actor_management()
    elif selected_page == "Settings":
        render_system_settings()
    elif selected_page == "Login / Activity Monitoring":
        render_activity_log()
    elif selected_page == "System & Model Status":
        render_system_model_status()

elif role == "Actor":
    if selected_page == "Interactive Map Check":
        render_heat_risk_map()
    elif selected_page == "Heat Risk Analysis":
        render_heat_risk_zones()
        render_dashboard_statistics()
    elif selected_page == "LST / NDVI / NDBI / NDWI Check":
        render_environmental_analysis()
    elif selected_page == "ML Prediction":
        render_ml_prediction()
    elif selected_page == "Activity Log":
        render_activity_log()

elif role == "Analyst":
    if selected_page == "Heatmap Analysis":
        render_heat_risk_map()
    elif selected_page == "Area / Grid Details":
        render_area_grid_details()
    elif selected_page == "ML Model Results":
        render_ml_analysis()
    elif selected_page == "Environmental Data Analysis":
        render_environmental_analysis()
    elif selected_page == "Reports / Visualization":
        render_reports_visualization()

elif role == "Viewer":
    if selected_page == "Login / Account":
        render_login_account()
    elif selected_page == "Interactive Map View":
        render_heat_risk_map()
    elif selected_page == "Heat Risk Zones":
        render_heat_risk_zones()
    elif selected_page == "Basic Area Information":
        render_area_grid_details()
    elif selected_page == "Dashboard Statistics":
        render_dashboard_statistics()

log_activity("Page Access", f"Role={role}; Page={selected_page}")
"""
Theme manager for DHAKA HEATMAP.

This file is intentionally separate from app.py/app_complete so the
Light/Dark appearance can be maintained independently.
"""

import streamlit as st


THEME_OPTIONS = {
    "☀️ Light Mode": "light",
    "🌙 Dark Mode": "dark",
}


def _inject_light_theme():
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #ffffff !important;
            color: #1f2937 !important;
        }

        [data-testid="stHeader"] {
            background: #ffffff !important;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc !important;
            color: #1f2937 !important;
        }

        [data-testid="stSidebarContent"] {
            background: #f8fafc !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stText"],
        label,
        p,
        span,
        h1, h2, h3, h4, h5, h6 {
            color: #1f2937;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"],
        div[data-baseweb="textarea"] {
            background: #ffffff !important;
            color: #1f2937 !important;
        }

        input,
        textarea {
            color: #1f2937 !important;
            background: #ffffff !important;
        }

        button {
            color: #1f2937;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_dark_theme():
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #0e1117 !important;
            color: #f5f7fa !important;
        }

        [data-testid="stHeader"] {
            background: #0e1117 !important;
        }

        [data-testid="stSidebar"] {
            background: #151a21 !important;
            color: #f5f7fa !important;
            border-right-color: #30363d !important;
        }

        [data-testid="stSidebarContent"] {
            background: #151a21 !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stText"],
        label,
        p,
        span,
        h1, h2, h3, h4, h5, h6 {
            color: #f5f7fa;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"],
        div[data-baseweb="textarea"] {
            background: #20262e !important;
            color: #f5f7fa !important;
            border-color: #3a424d !important;
        }

        input,
        textarea {
            color: #f5f7fa !important;
            background: #20262e !important;
            caret-color: #f5f7fa !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #9da7b3 !important;
        }

        button {
            color: #f5f7fa;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: #f5f7fa !important;
        }

        [data-testid="stDataFrame"] {
            border-color: #30363d !important;
        }

        hr {
            border-color: #30363d !important;
        }

        /* Keep common Streamlit cards/readouts readable. */
        [data-testid="stAlert"],
        [data-testid="stExpander"] {
            background: #171d24 !important;
            color: #f5f7fa !important;
            border-color: #30363d !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme(theme: str) -> None:
    """Apply the selected application theme."""
    if theme == "dark":
        _inject_dark_theme()
    else:
        _inject_light_theme()


def render_theme_selector(key: str = "app_theme") -> str:
    """
    Render a Light/Dark mode selector.

    Intended to be called inside the app's sidebar drawer.
    The selected theme is stored in session_state.
    """
    if "app_theme" not in st.session_state:
        st.session_state.app_theme = "light"

    labels = list(THEME_OPTIONS.keys())
    current_value = st.session_state.app_theme
    current_label = next(
        (label for label, value in THEME_OPTIONS.items() if value == current_value),
        labels[0],
    )

    selected_label = st.selectbox(
        "Appearance",
        labels,
        index=labels.index(current_label),
        key=key,
    )

    selected_theme = THEME_OPTIONS[selected_label]
    st.session_state.app_theme = selected_theme
    apply_theme(selected_theme)

    return selected_theme
