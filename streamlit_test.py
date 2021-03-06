import streamlit as st
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go

from utils import (
    filter_since_year, filter_by_wind_types, pie_distribution,
    get_color_dict, high_correlation
)


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


def get_sub_df(key: str, value: str, from_df=None):
    if from_df is None:
        from_df = final_df
    sub_df = from_df[from_df[key] == value]
    return sub_df


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

    st.markdown('## Pitch type distribution')
    st.write(pie_distribution(counts=pitch_type_count))

    # st.markdown('## Strike type distribution')
    # st.write(pie_distribution(counts=pitch_type_count))

    st.markdown('---\n## Pitch position scatter')
    strike_zone_distribution(df, atbat_df, targets=['pitch_type', 'code', 'event'])
    high_correlation(df, atbat_df)


def batter_page():
    name = st.sidebar.selectbox(
        'Which batter do you want to see?',
        final_df['Batters Name'].value_counts().index[:30])
    df = get_sub_df('Batters Name', name)
    atbat_df = get_sub_df('Batters Name', name, from_df=final_atbat_df)

    # Filtering
    st.sidebar.markdown('## Filtering')
    df, atbat_df = filter_since_year(df, atbat_df)
    df, atbat_df = filter_by_wind_types(df, atbat_df)

    st.markdown('---\n### Strike event distribution')
    event_counts = atbat_df['event'].value_counts()
    st.write(pie_distribution(counts=event_counts))
    code_counts = atbat_df['code'].value_counts()
    st.write(pie_distribution(counts=code_counts))

    st.markdown('---\n## Strike zone')
    strike_zone_distribution(df, atbat_df, targets=['code', 'event'])
    high_correlation(df, atbat_df)


def strike_zone_distribution(df, atbat_df, targets):
    def select_by_and_draw(df_, key, size=5):
        availible_types = list(df_[key].value_counts().index)
        st.markdown(f'### Select the {key} to show')
        selected_types = st.multiselect('', availible_types, default=availible_types[:2])
        type_to_color = get_color_dict(availible_types)
        fig = scatter_zone_on_selected_types(df_, key, selected_types, type_to_color, size=size)
        st.write(fig)

    if 'pitch_type' in targets:
        select_by_and_draw(df, 'pitch_type', size=3)
    if 'code' in targets:
        select_by_and_draw(df, 'code', size=3)
    if 'event' in targets:
        select_by_and_draw(atbat_df, 'event')


def scatter_zone_on_selected_types(df, key: str, selected_types: list, type_to_color: dict, size=5):
    data = []
    for _type in selected_types:
        color = type_to_color[_type]
        trace = go.Scatter(
            x=df.px[df[key] == _type],
            y=df.pz[df[key] == _type],
            name=_type,
            mode='markers',
            marker=dict(size=size, color=color, line=dict(width=2, color=color)))
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
