"""
Team Performance Heatmap Page.
Shows feature scores across circuit archetypes for each team.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


CLUSTER_NAMES = {
    0: 'High-Speed',
    1: 'Technical',
    2: 'Mixed',
    3: 'Street',
    4: 'Purpose-Built',
}


def render(data_dir: Path):
    st.header('Team Performance Heatmap')

    team_file = data_dir / 'legacy_team_features.parquet'

    if not team_file.exists():
        st.warning('No team feature data found. Run the pipeline first.')
        _show_demo_heatmap()
        return

    team_df = pd.read_parquet(team_file)

    # Season filter
    seasons = sorted(team_df['season'].unique())
    selected_season = st.selectbox('Season', seasons, index=len(seasons) - 1)
    season_data = team_df[team_df['season'] == selected_season]

    # Feature selector
    feature_options = [
        'race_pace_gap_pct', 'quali_pace_gap_pct', 'deg_rate_medium',
        'corner_speed_vs_straight_speed', 'high_speed_corner_grip',
        'brake_point_consistency', 'pu_vmax_avg',
    ]
    available_features = [f for f in feature_options if f in season_data.columns]
    selected_feature = st.selectbox('Feature', available_features,
                                      index=0 if available_features else 0)

    if not available_features:
        st.warning('No features available for visualization')
        return

    # Build heatmap: teams vs circuit clusters
    if 'circuit_cluster' in season_data.columns:
        st.subheader(f'{selected_feature} by Team and Circuit Type')

        pivot = season_data.pivot_table(
            values=selected_feature,
            index='team',
            columns='circuit_cluster',
            aggfunc='mean',
        )

        # Rename columns to cluster names
        pivot.columns = [CLUSTER_NAMES.get(int(c), f'Cluster {c}')
                         for c in pivot.columns]

        fig = px.imshow(
            pivot,
            color_continuous_scale='RdYlGn_r',
            labels={'color': selected_feature},
            aspect='auto',
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Team comparison across circuits
    st.subheader('Team Performance Across Circuits')

    teams = sorted(season_data['team'].unique())
    selected_teams = st.multiselect(
        'Select teams to compare', teams,
        default=teams[:5] if len(teams) >= 5 else teams)

    if selected_teams:
        compare_data = season_data[season_data['team'].isin(selected_teams)]

        fig = px.line(
            compare_data.sort_values('round'),
            x='circuit', y=selected_feature,
            color='team',
            markers=True,
            labels={'circuit': 'Circuit', selected_feature: selected_feature},
        )
        fig.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Season averages
    st.subheader('Season Average Performance')
    season_avg = (
        season_data.groupby('team')[selected_feature]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        season_avg, x=selected_feature, y='team',
        orientation='h',
        color=selected_feature,
        color_continuous_scale='RdYlGn_r',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_demo_heatmap():
    """Demo heatmap when no data available."""
    st.info('Run the feature pipeline to see real performance data.')

    teams = ['Red Bull', 'Ferrari', 'McLaren', 'Mercedes', 'Aston Martin']
    clusters = ['High-Speed', 'Technical', 'Mixed', 'Street']

    np.random.seed(42)
    data = np.random.rand(len(teams), len(clusters)) * 1.5

    fig = px.imshow(
        data,
        x=clusters, y=teams,
        color_continuous_scale='RdYlGn_r',
        labels={'color': 'Pace Gap (%)'},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
