"""
F1 Telemetry AI — Streamlit Dashboard
Multi-page app loading pre-computed predictions from parquet files.
No live model inference — predictions updated by pipeline script.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='F1 Telemetry AI',
    page_icon='🏎️',
    layout='wide',
    initial_sidebar_state='expanded',
)

DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'


def main():
    st.sidebar.title('F1 Telemetry AI')
    st.sidebar.markdown('AI/ML race performance analysis and prediction')

    page = st.sidebar.radio(
        'Navigate',
        ['Race Prediction', 'Circuit Profile', 'Team Heatmap', 'Reliability Tracker'],
    )

    if page == 'Race Prediction':
        from .pages.race_prediction import render
        render(DATA_DIR)
    elif page == 'Circuit Profile':
        from .pages.circuit_profile import render
        render(DATA_DIR)
    elif page == 'Team Heatmap':
        from .pages.team_heatmap import render
        render(DATA_DIR)
    elif page == 'Reliability Tracker':
        from .pages.reliability_tracker import render
        render(DATA_DIR)

    st.sidebar.markdown('---')
    st.sidebar.markdown(
        'Data: FastF1 / OpenF1 / Ergast  \n'
        'Model: XGBoost with SHAP  \n'
        'Built with Streamlit'
    )


if __name__ == '__main__':
    main()
