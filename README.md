# MLB data streamlit visualzation

## Introduction

In competitive sports like baseball, **personalized analysis on the opponents or your team
members** is as important as the analysis over all players and matches. For example, coaches
can decide strategies or the training content if they have a tool to know the playing style of their
opponents before matches.

Therefore, we developed a web service which provides an **easy-to-use interactive interface**
for non-programmers to access personalized analysis of a player. This service presents overall
statistics of a player and automatically find notable correlation or association among the
pitch/strike events and other conditions.

Try it here: http://140.112.29.149:8501/
(will be closed once the final project is finished)

To run the service: `streamlit run streamlit_test.py`

## Tools

- Streamlit: This is a very convenient python framework to develop ML web tools. It’s
    compatible to other python packages of data science ecosystem. Visit their website if
    you want more information: ​https://www.streamlit.io/
- Analysis tools: numpy, pandas, mlextend
- Visualization: matplotlib, seaborn, plotly

## Interface

![](https://i.imgur.com/PTRUhQ4.png)

On the left side, users can choose the analysis type (pitcher or batter), which player, and some
filtering conditions such as the starting year of data to be used. And, the automatic analyzation
results will be shown on the right.

## Provided analysis

![](https://i.imgur.com/RfhYI4S.png)

We provide some basic distribution analysis like pitch types and strike events at the top of our
analysis, therefore users can quickly get a rough concept of that player.

![](https://i.imgur.com/g53vYdO.png)

Besides, we also provide some spatial distribution visualization to help users dig into the
tendency of the analyzed player. For instance, users can compare the distribution of two kinds
of pitch types from that pitcher like the example above.

![](https://i.imgur.com/NxohPog.png)

Since our service is interactive, users can also choose an arbitrary number of types to compare.
Other spatial distribution like strike zone are also provided.

## Correlation/association mining

![](https://i.imgur.com/AxK2Oil.png)

Besides those distribution visualization, we provide notable correlation and association. We first
choose interesting outcome columns and possible cause columns (e.g. ball spin rate, pitch type,
or wind direction). Then, we calculate the correlation or association between each pair of
outcome columns and cause columns, filter out those with small absolute values, and show the
notable correlation/association.

The codes of this service are available here:
https://github.com/Nash2325138/MLB-data-streamlit-visualzation-practice


