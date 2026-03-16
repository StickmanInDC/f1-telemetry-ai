"""
Race Prediction Page.
Shows predicted pace gaps for upcoming race, team rankings,
and SHAP explanations for each prediction.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def render(data_dir: Path):
    st.header('Race Prediction')

    # Load predictions
    pred_file = data_dir / 'predictions_latest.parquet'
    shap_file = data_dir / 'shap_importance.parquet'

    if not pred_file.exists():
        st.warning(
            'No predictions found. Run the pipeline first:\n\n'
            '```\npython -m src.features.pipeline\n'
            'python -m src.models.legacy_model\n```'
        )
        _show_demo_predictions()
        return

    predictions = pd.read_parquet(pred_file)

    # Circuit selector
    circuits = predictions['circuit'].unique()
    selected_circuit = st.selectbox('Select Circuit', circuits,
                                      index=len(circuits) - 1)

    circuit_preds = predictions[predictions['circuit'] == selected_circuit].copy()
    circuit_preds = circuit_preds.sort_values('predicted_pace_gap_pct')

    # Prediction table
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader('Predicted Team Rankings')
        display_df = circuit_preds[['team', 'predicted_pace_gap_pct']].reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.columns = ['Team', 'Gap to Leader (%)']
        st.dataframe(display_df, use_container_width=True)

    with col2:
        st.subheader('Pace Gap Visualization')
        fig = px.bar(
            circuit_preds,
            x='predicted_pace_gap_pct',
            y='team',
            orientation='h',
            color='predicted_pace_gap_pct',
            color_continuous_scale='RdYlGn_r',
            labels={'predicted_pace_gap_pct': 'Gap to Leader (%)', 'team': 'Team'},
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP explanation
    if shap_file.exists():
        st.subheader('Why does the model predict this?')
        shap_df = pd.read_parquet(shap_file)
        fig_shap = px.bar(
            shap_df.head(10),
            x='mean_abs_shap',
            y='feature',
            orientation='h',
            labels={'mean_abs_shap': 'Feature Importance (|SHAP|)', 'feature': ''},
            color='mean_abs_shap',
            color_continuous_scale='Blues',
        )
        fig_shap.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Model accuracy metrics
    metrics_file = data_dir / 'model_metrics.parquet'
    if metrics_file.exists():
        st.subheader('Model Accuracy')
        metrics = pd.read_parquet(metrics_file)
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'mae' in metrics.columns:
                st.metric('MAE (% gap)', f"{metrics['mae'].iloc[-1]:.3f}")
        with col2:
            if 'top3_accuracy' in metrics.columns:
                st.metric('Top-3 Accuracy', f"{metrics['top3_accuracy'].iloc[-1]:.0%}")
        with col3:
            if 'test_season' in metrics.columns:
                st.metric('Test Season', int(metrics['test_season'].iloc[-1]))


def _show_demo_predictions():
    """Show demo predictions when no real data is available."""
    st.subheader('Demo Predictions (Sample Data)')
    st.info('These are placeholder predictions. Run the pipeline to generate real predictions.')

    demo_data = pd.DataFrame({
        'team': ['Red Bull Racing', 'Ferrari', 'McLaren', 'Mercedes',
                 'Aston Martin', 'Alpine', 'Williams', 'RB',
                 'Haas F1 Team', 'Sauber'],
        'predicted_pace_gap_pct': [0.0, 0.15, 0.22, 0.38, 0.55,
                                     0.72, 0.85, 0.91, 1.05, 1.15],
        'circuit': ['Demo Circuit'] * 10,
    })

    col1, col2 = st.columns([2, 3])

    with col1:
        display = demo_data[['team', 'predicted_pace_gap_pct']].copy()
        display.index = range(1, len(display) + 1)
        display.columns = ['Team', 'Gap to Leader (%)']
        st.dataframe(display, use_container_width=True)

    with col2:
        fig = px.bar(
            demo_data,
            x='predicted_pace_gap_pct',
            y='team',
            orientation='h',
            color='predicted_pace_gap_pct',
            color_continuous_scale='RdYlGn_r',
            labels={'predicted_pace_gap_pct': 'Gap to Leader (%)', 'team': 'Team'},
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
