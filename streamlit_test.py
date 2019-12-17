import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.offline as py
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly import tools
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import random


def random_colors(number_of_colors):
    color = [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for i in range(number_of_colors)
    ]
    return color


@st.cache
def get_df(path):
    df = pd.read_csv(path)
    _winds = [v.split(' mph') for v in df.wind.values]
    df['wind_number'] = [int(w[0]) for w in _winds]
    df['wind_types'] = [w[1][2:] for w in _winds]
    df['year'] = [int(v[:4]) for v in df.date.values]
    df['mounth'] = [int(v[5:7]) for v in df.date.values]
    df_ab = df.groupby('ab_id').last()
    return df, df_ab


@st.cache
def get_random_rgb_colors(k):
    return [list(random.choices(range(256), k=3)) for _ in range(k)]


def get_sub_df(key: str, value: str, from_df=None):
    if from_df is None:
        from_df = final_df
    sub_df = from_df[from_df[key] == value]
    return sub_df


def pie_distribution(series=None, counts=None):
    if series is not None:
        event_counts = series.value_counts()
    elif counts is not None:
        event_counts = counts
    else:
        raise ValueError
    fig = px.pie(values=event_counts.values, names=event_counts.index)
    fig = fig.update_xaxes(categoryorder='total descending')
    fig = fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def filter_since_year(df, atbat_df):
    years = sorted([int(y) for y in set(df.year.values)])
    since_year = st.sidebar.slider('Since which year', years[0], years[-1], years[0])
    out_df = df[df['year'] >= since_year]
    out_atbat_df = atbat_df[atbat_df['year'] >= since_year]
    return out_df, out_atbat_df


def filter_by_wind_types(df, atbat_df):
    all_wind_types = list(df['wind_types'].value_counts().index)
    selected_wind_types = st.sidebar.multiselect(
        'Wind directions', all_wind_types, default=all_wind_types)
    out_df = df[df['wind_types'].isin(selected_wind_types)]
    out_atbat_df = atbat_df[atbat_df['wind_types'].isin(selected_wind_types)]
    return out_df, out_atbat_df


def get_color_dict(events):
    pass


def pitcher_page():
    name = st.sidebar.selectbox(
        'Which pitcher do you want to see?',
        final_df['Pitchers Name'].value_counts().index[:30])
    df = get_sub_df("Pitchers Name", name)
    atbat_df = get_sub_df("Pitchers Name", name, from_df=final_atbat_df)

    # Filtering
    st.sidebar.markdown('## Filtering')
    df, atbat_df = filter_since_year(df, atbat_df)
    df, atbat_df = filter_by_wind_types(df, atbat_df)

    pitch_type_count = df.pitch_type.value_counts()
    availible_pitch_types = list(pitch_type_count.index)

    st.markdown('## Pitch type distribution')
    st.write(pie_distribution(counts=pitch_type_count))

    # st.markdown('## Strike type distribution')
    # st.write(pie_distribution(counts=pitch_type_count))

    st.markdown('---\n## Pitch position scatter')
    selected_pitch_types = st.multiselect(
        'Select the pitch types to show',
        availible_pitch_types,
        default=availible_pitch_types[:2])

    all_pitch_types = list(final_df.pitch_type.value_counts().index)
    colors = get_random_rgb_colors(len(all_pitch_types))
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, .4)' for c in colors]
    type_to_color = {t: c for t, c in zip(all_pitch_types, colors)}

    fig = scatter_zone_on_selected_types(df, 'pitch_type', selected_pitch_types, type_to_color)
    st.write(fig)


def batter_page():
    name = st.sidebar.selectbox(
        'Which batter do you want to see?',
        final_df['Batters Name'].value_counts().index[:30])
    batter_df = get_sub_df('Batters Name', name)
    batter_atbat_df = get_sub_df('Batters Name', name, from_df=final_atbat_df)

    # Filtering
    st.sidebar.markdown('## Filtering')
    batter_df, batter_atbat_df = filter_since_year(batter_df, batter_atbat_df)
    batter_df, batter_atbat_df = filter_by_wind_types(batter_df, batter_atbat_df)

    st.markdown('### Strike event distribution')
    # code_counts = batter_df['code'].value_counts()
    event_counts = batter_atbat_df['event'].value_counts()
    st.write(pie_distribution(counts=event_counts))
    code_counts = batter_atbat_df['code'].value_counts()
    st.write(pie_distribution(counts=code_counts))

    st.markdown('---\n## Strike zone')

#     # Select codes
#     availible_event_types = list(event_counts.index)
#     selected_event_types = st.multiselect('Select the event types to show', availible_event_types,
#                                           default=availible_event_types[:2])

    # Select events
    availible_event_types = list(event_counts.index)
    selected_event_types = st.multiselect('Select the event types to show',
                                          availible_event_types, default=availible_event_types[:2])

    # Assign color to each event
    all_event_types = list(final_df.event.value_counts().index)
    colors = get_random_rgb_colors(len(all_event_types))
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, .4)' for c in colors]
    type_to_color = {t: c for t, c in zip(all_event_types, colors)}

    fig = scatter_zone_on_selected_types(batter_atbat_df, 'event', selected_event_types, type_to_color)
    st.write(fig)


def scatter_zone_on_selected_types(df, key: str, selected_types: list, type_to_color: dict):
    data = []
    for _type in selected_types:
        color = type_to_color[_type]
        trace = go.Scatter(
            x=df.px[df[key] == _type],
            y=df.pz[df[key] == _type],
            name=_type,
            mode='markers',
            marker=dict(size=5, color=color, line=dict(width=2, color=color)))
        data.append(trace)
    fig = go.Figure(data=data)
    return fig


@st.cache
def get_final_basic_measure(columns):
    return final_df[columns].mean(axis=0), final_df[columns].std(axis=0)


NUMERIC_COLUMNS = ['px', 'pz', 'start_speed', 'end_speed', 'spin_rate', 'spin_dir',
                   'break_angle', 'break_length', 'break_y']

final_df, final_atbat_df = get_df("../input/all_joined.csv.zip")
final_mean, final_std = get_final_basic_measure(NUMERIC_COLUMNS)

st.sidebar.markdown('# Analysis Target')
service_names = ('Pitcher', 'Batter')
service_type = st.sidebar.selectbox(
    label="Type", options=service_names, index=0)

if service_type == service_names[0]:
    pitcher_page()
elif service_type == service_names[1]:
    batter_page()
