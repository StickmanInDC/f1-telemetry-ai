"""
Reliability Tracker Page.
DNF history by failure mode, rolling trend chart,
and PU supplier reliability comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


SYSTEMIC_CATEGORIES = {'power_unit', 'electrical', 'hydraulics', 'mechanical', 'chassis'}

PU_SUPPLIER_MAP = {
    'Ferrari': 'Ferrari PU',
    'Haas F1 Team': 'Ferrari PU',
    'Haas': 'Ferrari PU',
    'MoneyGram Haas F1 Team': 'Ferrari PU',
    'Cadillac': 'Ferrari PU',
    'Mercedes': 'Mercedes PU',
    'McLaren': 'Mercedes PU',
    'McLaren F1 Team': 'Mercedes PU',
    'Williams': 'Mercedes PU',
    'Williams Racing': 'Mercedes PU',
    'Alpine': 'Mercedes PU',
    'BWT Alpine F1 Team': 'Mercedes PU',
    'Red Bull Racing': 'Red Bull/Ford PU',
    'Oracle Red Bull Racing': 'Red Bull/Ford PU',
    'RB': 'Red Bull/Ford PU',
    'Racing Bulls': 'Red Bull/Ford PU',
    'Visa Cash App RB F1 Team': 'Red Bull/Ford PU',
    'Aston Martin': 'Honda PU',
    'Aston Martin Aramco F1 Team': 'Honda PU',
    'Sauber': 'Audi PU',
    'Kick Sauber': 'Audi PU',
    'Stake F1 Team Kick Sauber': 'Audi PU',
}


def render(data_dir: Path):
    st.header('Reliability Tracker')

    results_file = data_dir / 'results_with_dnf.parquet'

    if not results_file.exists():
        st.warning('No results data found. Run the pipeline first.')
        _show_demo_reliability()
        return

    results_df = pd.read_parquet(results_file)

    # Season filter
    seasons = sorted(results_df['season'].unique())
    selected_season = st.selectbox('Season', seasons, index=len(seasons) - 1)
    season_data = results_df[results_df['season'] == selected_season]

    # DNF overview
    st.subheader('DNF Classification Summary')
    _render_dnf_summary(season_data)

    # Team reliability scores
    st.subheader('Team Reliability Scores')
    _render_team_reliability(season_data)

    # PU supplier comparison
    st.subheader('PU Supplier Reliability')
    _render_pu_reliability(season_data)

    # Rolling trend
    st.subheader('Reliability Trend (Rolling 5 Races)')
    _render_reliability_trend(season_data)

    # Detailed DNF log
    st.subheader('DNF Log')
    dnf_data = season_data[season_data['dnf_category'] != 'finished']
    if not dnf_data.empty:
        display_cols = ['round', 'circuit_name', 'constructor_name',
                         'driver_code', 'status', 'dnf_category', 'laps_completed']
        available_cols = [c for c in display_cols if c in dnf_data.columns]
        st.dataframe(dnf_data[available_cols].sort_values('round'),
                      use_container_width=True)
    else:
        st.info('No DNFs recorded for this season')


def _render_dnf_summary(season_data: pd.DataFrame):
    """Pie chart of DNF categories."""
    if 'dnf_category' not in season_data.columns:
        return

    dnf_only = season_data[season_data['dnf_category'] != 'finished']
    if dnf_only.empty:
        st.info('No DNFs in selected season')
        return

    category_counts = dnf_only['dnf_category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    col1, col2 = st.columns([1, 2])
    with col1:
        total_starts = len(season_data)
        total_dnfs = len(dnf_only)
        systemic = dnf_only[dnf_only['dnf_category'].isin(SYSTEMIC_CATEGORIES)]

        st.metric('Total DNFs', total_dnfs)
        st.metric('Systemic Failures', len(systemic))
        st.metric('DNF Rate', f"{total_dnfs / total_starts * 100:.1f}%")

    with col2:
        fig = px.pie(category_counts, values='Count', names='Category',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def _render_team_reliability(season_data: pd.DataFrame):
    """Bar chart of completion rate per team."""
    team_col = 'constructor_name' if 'constructor_name' in season_data.columns else 'team'

    if 'total_laps' in season_data.columns:
        team_completion = (
            season_data.groupby(team_col)
            .apply(lambda g: (g['laps_completed'] / g['total_laps'].clip(lower=1)).mean())
            .sort_values(ascending=False)
            .reset_index(name='completion_rate')
        )
    else:
        team_completion = (
            season_data.groupby(team_col)
            .apply(lambda g: (g['dnf_category'] == 'finished').mean())
            .sort_values(ascending=False)
            .reset_index(name='completion_rate')
        )

    fig = px.bar(
        team_completion,
        x='completion_rate', y=team_col,
        orientation='h',
        color='completion_rate',
        color_continuous_scale='RdYlGn',
        labels={'completion_rate': 'Completion Rate', team_col: 'Team'},
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pu_reliability(season_data: pd.DataFrame):
    """Compare reliability across PU suppliers."""
    team_col = 'constructor_name' if 'constructor_name' in season_data.columns else 'team'
    df = season_data.copy()
    df['pu_supplier'] = df[team_col].map(PU_SUPPLIER_MAP).fillna('Unknown')

    systemic = df[df['dnf_category'].isin(SYSTEMIC_CATEGORIES)]
    supplier_dnfs = systemic.groupby('pu_supplier').size().reset_index(name='systemic_dnfs')
    supplier_starts = df.groupby('pu_supplier').size().reset_index(name='starts')
    supplier_data = supplier_starts.merge(supplier_dnfs, on='pu_supplier', how='left')
    supplier_data['systemic_dnfs'] = supplier_data['systemic_dnfs'].fillna(0)
    supplier_data['systemic_dnf_rate'] = supplier_data['systemic_dnfs'] / supplier_data['starts']
    supplier_data = supplier_data.sort_values('systemic_dnf_rate')

    fig = px.bar(
        supplier_data,
        x='systemic_dnf_rate', y='pu_supplier',
        orientation='h',
        color='systemic_dnf_rate',
        color_continuous_scale='RdYlGn_r',
        labels={'systemic_dnf_rate': 'Systemic DNF Rate', 'pu_supplier': 'PU Supplier'},
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_reliability_trend(season_data: pd.DataFrame):
    """Rolling completion rate trend over the season."""
    team_col = 'constructor_name' if 'constructor_name' in season_data.columns else 'team'

    if 'total_laps' not in season_data.columns:
        season_data = season_data.copy()
        season_data['total_laps'] = season_data.groupby('round')['laps_completed'].transform('max')

    per_race_team = (
        season_data.groupby([team_col, 'round'])
        .apply(lambda g: (g['laps_completed'] / g['total_laps'].clip(lower=1)).mean())
        .reset_index(name='completion_rate')
    )

    # Rolling 5-race average
    trend_data = []
    for team in per_race_team[team_col].unique():
        team_data = per_race_team[per_race_team[team_col] == team].sort_values('round')
        team_data['rolling_completion'] = team_data['completion_rate'].rolling(5, min_periods=1).mean()
        trend_data.append(team_data)

    if trend_data:
        trend_df = pd.concat(trend_data)
        fig = px.line(
            trend_df, x='round', y='rolling_completion',
            color=team_col,
            labels={'round': 'Race Round', 'rolling_completion': 'Rolling Completion Rate'},
            markers=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _show_demo_reliability():
    """Demo view when no data available."""
    st.info('Run the feature pipeline to see real reliability data.')

    demo = pd.DataFrame({
        'Team': ['Honda PU (Aston Martin)', 'Audi PU (Sauber)', 'Red Bull/Ford PU',
                 'Ferrari PU', 'Mercedes PU'],
        'Systemic DNF Rate': [0.35, 0.25, 0.15, 0.05, 0.03],
        'Status': ['Crisis', 'Concerning', 'Monitoring', 'Stable', 'Stable'],
    })

    fig = px.bar(
        demo, x='Systemic DNF Rate', y='Team',
        orientation='h', color='Status',
        color_discrete_map={'Crisis': 'red', 'Concerning': 'orange',
                             'Monitoring': 'yellow', 'Stable': 'green'},
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
    st.plotly_chart(fig, use_container_width=True)
