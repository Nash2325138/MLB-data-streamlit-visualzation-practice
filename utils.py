from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend import frequent_patterns
import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import random


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


def get_color_dict(all_types):
    colors = get_random_rgb_colors(len(all_types))
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, .4)' for c in colors]
    type_to_color = {t: c for t, c in zip(all_types, colors)}
    return type_to_color


@st.cache
def get_random_rgb_colors(k):
    out = []
    while len(out) < k:
        c = list(random.choices(range(256), k=3))
        if len(out) >= 1:
            if np.abs(np.array(c) - np.array(out[-1])).sum() <= 50:
                continue
        out.append(c)
    return out


def random_colors(number_of_colors):
    color = [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for i in range(number_of_colors)
    ]
    return color


def high_correlation(df, atbat_df):
    continues_causes = ['start_speed', 'end_speed', 'spin_rate', 'spin_dir',
                        'break_angle', 'break_length', 'break_y', 'inning']
    categorical_causes = ['pitch_type', 'wind_types']

    # Association with code
    def calc_and_show_association(df_, key):
        tmp = pd.concat([
            pd.get_dummies(df_[categorical_causes]),
            pd.get_dummies(df_[[key]])
        ], axis=1)
        frequent_itemsets = apriori(tmp, min_support=0.05, use_colnames=True)
        associations = association_rules(
            frequent_itemsets, metric='confidence', min_threshold=0.2
        ).sort_values('confidence', ascending=False)
        associations = associations[[
            all([t.startswith(f'{key}_') for t in v])
            for v in associations['consequents']
        ]]
        associations['antecedents'] = [set(a) for a in associations['antecedents']]
        associations['consequents'] = [set(a) for a in associations['consequents']]
        del associations['leverage']
        del associations['conviction']
        st.table(associations)

    st.markdown('---\n ### Notable association')
    calc_and_show_association(df, 'code')
    calc_and_show_association(atbat_df, 'event')

    def calc_and_show_correlation(df_, key):
        out_one_hot = pd.get_dummies(df_[key])
        continues_cause_df = df_[continues_causes]
        corrs = pd.concat([continues_cause_df, out_one_hot], axis=1).corr()
        corrs = corrs.loc[continues_cause_df.columns, out_one_hot.columns]
        corrs_list = [
            (r, c, corrs.loc[r, c])
            for r in corrs.index
            for c in corrs.columns
        ]
        corrs_list = [(r, c, v) for r, c, v in corrs_list
                      if not np.isnan(v) and abs(v) > 0.1]
        corrs_list = sorted(corrs_list, key=lambda a: abs(a[2]))[::-1]
        if len(corrs_list) > 0:
            correlation_df = pd.DataFrame(corrs_list)
            correlation_df.columns = ['attribute', 'outcome', 'correlation coef']
            st.write(correlation_df)

    st.markdown('---\n ### Notable correlation')
    calc_and_show_correlation(df, 'code')
    calc_and_show_correlation(atbat_df, 'event')
