"""
Circuit Profile Page.
Shows track cluster assignment, key characterization features,
and energy richness for each circuit.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


CLUSTER_NAMES = {
    0: 'High-Speed / Power',
    1: 'Technical / Mechanical',
    2: 'Mixed Character',
    3: 'Street / Bumpy',
    4: 'Modern Purpose-Built',
}

CLUSTER_COLORS = {
    0: '#FF4136',
    1: '#2ECC40',
    2: '#FF851B',
    3: '#B10DC9',
    4: '#0074D9',
}


def render(data_dir: Path):
    st.header('Circuit Profile')

    track_file = data_dir / 'legacy_track_features.parquet'

    if not track_file.exists():
        st.warning('No track feature data found. Run the pipeline first.')
        _show_demo_profile()
        return

    track_df = pd.read_parquet(track_file)

    # Aggregate by circuit (latest season)
    latest = track_df.groupby('circuit').last().reset_index()

    # Circuit selector
    circuits = sorted(latest['circuit'].unique())
    selected = st.selectbox('Select Circuit', circuits)

    circuit_data = latest[latest['circuit'] == selected].iloc[0]

    # Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cluster_id = int(circuit_data.get('circuit_cluster', -1))
        cluster_name = CLUSTER_NAMES.get(cluster_id, 'Unknown')
        st.metric('Circuit Archetype', cluster_name)
    with col2:
        st.metric('Full Throttle %',
                   f"{circuit_data.get('pct_full_throttle', 0) * 100:.1f}%")
    with col3:
        st.metric('Street Circuit',
                   'Yes' if circuit_data.get('is_street_circuit', 0) else 'No')
    with col4:
        alt = circuit_data.get('altitude_m', 0)
        st.metric('Altitude', f"{alt:.0f}m" if not pd.isna(alt) else 'N/A')

    # Feature radar chart
    st.subheader('Circuit Characteristics')
    _render_radar_chart(circuit_data)

    # All circuits comparison
    st.subheader('All Circuits by Cluster')
    _render_cluster_scatter(latest)

    # Circuit comparison
    st.subheader('Compare Circuits')
    compare = st.multiselect('Select circuits to compare', circuits,
                                default=circuits[:3] if len(circuits) >= 3 else circuits)
    if compare:
        compare_data = latest[latest['circuit'].isin(compare)]
        features_to_compare = ['pct_full_throttle', 'pct_heavy_braking',
                                'avg_corner_speed', 'num_slow_corners',
                                'energy_recovery_potential']
        available = [f for f in features_to_compare if f in compare_data.columns]
        if available:
            fig = px.bar(
                compare_data.melt(id_vars='circuit', value_vars=available),
                x='variable', y='value', color='circuit', barmode='group',
                labels={'variable': 'Feature', 'value': 'Value', 'circuit': 'Circuit'},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(circuit_data: pd.Series):
    """Render a radar chart of circuit characteristics."""
    features = {
        'Full Throttle': circuit_data.get('pct_full_throttle', 0),
        'Heavy Braking': circuit_data.get('pct_heavy_braking', 0),
        'Corner Speed': circuit_data.get('avg_corner_speed', 0) / 300,  # Normalize
        'Slow Corners': circuit_data.get('num_slow_corners', 0) / 10,
        'Fast Corners': circuit_data.get('num_fast_corners', 0) / 10,
        'Energy Recovery': min(1.0, circuit_data.get('energy_recovery_potential', 0) / 10),
    }

    fig = go.Figure(data=go.Scatterpolar(
        r=list(features.values()),
        theta=list(features.keys()),
        fill='toself',
        line_color='#FF4136',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cluster_scatter(track_df: pd.DataFrame):
    """Show all circuits colored by cluster."""
    if 'circuit_cluster' not in track_df.columns:
        st.info('No cluster data available')
        return

    df = track_df.copy()
    df['cluster_name'] = df['circuit_cluster'].map(
        lambda x: CLUSTER_NAMES.get(int(x), 'Unknown') if not pd.isna(x) else 'Unknown')

    x_col = 'pct_full_throttle' if 'pct_full_throttle' in df.columns else None
    y_col = 'avg_corner_speed' if 'avg_corner_speed' in df.columns else None

    if x_col and y_col:
        fig = px.scatter(
            df, x=x_col, y=y_col, color='cluster_name',
            hover_data=['circuit'],
            labels={x_col: 'Full Throttle %', y_col: 'Avg Corner Speed (km/h)'},
            text='circuit',
        )
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


def _show_demo_profile():
    """Show demo when no data available."""
    st.info('Run the feature pipeline to see real circuit profiles.')

    demo = pd.DataFrame({
        'Circuit': ['Monza', 'Monaco', 'Silverstone', 'Bahrain', 'Suzuka'],
        'Archetype': ['High-Speed', 'Technical', 'High-Speed', 'Mixed', 'Mixed'],
        'Full Throttle %': [78, 42, 68, 62, 64],
        'Slow Corners': [2, 8, 3, 4, 5],
    })
    st.dataframe(demo, use_container_width=True)
