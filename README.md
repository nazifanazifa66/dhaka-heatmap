# DHAKA HEATMAP — Streamlit Cloud Deployment

This folder contains the complete deployment package for the **DHAKA HEATMAP — Urban Heat Risk Monitoring System**.

## Included application features

- Full-page login
- Four role types: Admin, Actor, Analyst, Viewer
- Role-specific collapsible Streamlit sidebar drawer
- Interactive Dhaka heat-risk grid map
- Grid-level LST / NDVI / NDBI / NDWI information
- Heat-risk categories and statistics
- Linear Regression, Random Forest and XGBoost model analysis
- Interactive ML prediction
- Reports / visualization
- Activity monitoring
- Admin user management and system/model status

## Files

- `app.py` — Streamlit entrypoint
- `dhaka_heat_risk_data.csv` — heat-risk results
- `Dhaka_Heatmap_Grid_Features.csv` — source grid/environmental features and `system:index`
- `Dhaka_Heatmap_Grid_Final.geojson` — final 500 m Dhaka grid with `grid_id`
- `random_forest_model.joblib` — Random Forest model
- `linear_regression_model.joblib` — Linear Regression model
- `xgboost_model.json` — native XGBoost model
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server/browser configuration

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `dhaka-heatmap`.
2. Upload **all files in this folder**, including the model and data files.
3. Go to Streamlit Community Cloud and choose **Create app**.
4. Select the GitHub repository, branch, and `app.py` as the entrypoint.
5. In Advanced settings, use Python 3.12 unless you have a specific reason to use another supported version.
6. Deploy the app.
7. Open the resulting `*.streamlit.app` URL for the live presentation.

## Demo accounts

These are demo credentials currently embedded in the application code for the academic presentation build:

- Admin: `admin` / `admin123`
- Actor: `actor` / `actor123`
- Analyst: `analyst` / `analyst123`
- Viewer: `viewer` / `viewer123`

**Important:** These hard-coded accounts are suitable for an academic/demo deployment, not for a production public system. For production, move credentials to a secure authentication system or Streamlit secrets.

## Grid ID handling

The app does not assume that `dhaka_heat_risk_data.csv` contains `grid_id`. It creates the dashboard-level `grid_id` from `Dhaka_Heatmap_Grid_Features.csv` → `system:index`, matching the GeoJSON `properties.grid_id`. The supplied project files contain 6,272 rows/features and the feature/GeoJSON identifier sets match.
