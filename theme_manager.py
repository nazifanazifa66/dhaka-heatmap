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
