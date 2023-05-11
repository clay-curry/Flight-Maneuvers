from dataset import MANEUVERS

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_scenario_topdown(true_df, pred_df):
    """ Plot the scenario in 2D, with the true maneuver on the left and the predicted maneuver on the right.

    Args:

        true_df (pandas.DataFrame): A dataframe with the columns 'x', 'y', and 'maneuver'. The 'maneuver' column should be a string with one of the following values: 'takeoff', 'turn', 'line', 'orbit', 'landing'.

        pred_df (pandas.DataFrame): A dataframe with the columns 'takeoff', 'turn', 'line', 'orbit', 'landing', and 'maneuver'. The 'maneuver' column should be a string with one of the following values: 'takeoff', 'turn', 'line', 'orbit', 'landing'. The other columns should be a number between 0 and 1, indicating the probability that the maneuver is the corresponding column name.

    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("True Maneuver", "Pred Maneuver"))
    fig.add_trace(go.Scatter(x=true_df['x'], y=true_df['y'], showlegend=False, visible=False), row=1,col=1)
    fig.add_trace(go.Scatter(x=true_df['x'], y=true_df['y'], showlegend=False, visible=False), row=1,col=2)
    fig.update_traces(mode='markers', marker=dict(colorscale=[[0, 'rgb(150, 150, 250)'], [1, 'rgb(250, 0, 0)']], showscale=True))
    for s, c in zip(MANEUVERS, px.colors.qualitative.Plotly):
        t_mask = true_df['maneuver'] == s
        p_mask = pred_df['maneuver'] == s        
        fig.add_trace(go.Scatter(x=true_df['x'][t_mask], y=true_df['y'][t_mask], name=s, mode='markers', marker_color=c, showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=true_df['x'][p_mask.values], y=true_df['y'][p_mask.values], name=s, mode='markers', marker_color=c, showlegend=True), row=1,col=2)
    dropdown_menu = dict(buttons=list([dict(label=k, method="update",  args=[{"visible": [True, True] + [False] * 10, "marker.color": [(true_df['maneuver']==k).astype(float), pred_df[k]]}]) for k in MANEUVERS]
        ),  direction="down", showactive=True, xanchor="right", yanchor="top")
    
    fig.update_layout(updatemenus=[dropdown_menu], legend=dict(yanchor="bottom", y=0.30, xanchor="left", x=0))
    fig.show()
