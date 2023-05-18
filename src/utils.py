import torch
import numpy as np
import pandas as pd
from math import inf

from src import EXPERIMENT_PATH
from src.data_module import MANEUVERS

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model.resnet import ResNet
from src.data_module import FlightTrajectoryDataModule

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor



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

tokenize_maneuvers = np.vectorize(lambda id: MANEUVERS.index(id))

def sample_timeseries(trajectory, sampling_period=1, max_length=inf):
    """ loads and resamples a trajectory to a fixed sampling period and maximum length

    Args:
        trajectory: path to a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver
        sampling_period: the desired sampling period in seconds
        max_length: the maximum length of the resampled trajectory

    Returns:
        a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver
    """
    # load the trajectory and convert the 't' column to a DatetimeIndex
    trajectory = pd.read_csv(trajectory)

    # initialize resampled_trajectory with the first row of trajectory
    resampled_trajectory = trajectory.iloc[::sampling_period]

    # trim the trajectory to the desired maximum length
    if len(resampled_trajectory) > max_length:
        resampled_trajectory = resampled_trajectory.iloc[:max_length]

    resampled_trajectory = resampled_trajectory.reset_index(drop=True)
    return resampled_trajectory

def preprocess_trajectory(trajectory, target = None):
    """ computes the delta of a trajectory

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver

    Returns:
        a pandas dataframe with columns x, y, z, vx, xy, vz, dx, dy, dz, dvx, dvy, dvz, maneuver
    """
    maneuvers = tokenize_maneuvers(trajectory['maneuver'])
    trajectory = trajectory.drop('maneuver', axis=1).drop('t', axis=1)

    # initialize delta_trajectory with the first row of trajectory
    trajectory['dx'] = trajectory['x'].diff()
    trajectory['dy'] = trajectory['y'].diff()
    trajectory['dz'] = trajectory['z'].diff()
    trajectory['dvx'] = trajectory['vx'].diff()
    trajectory['dvy'] = trajectory['vy'].diff()
    trajectory['dvz'] = trajectory['vz'].diff()
    trajectory['dx'].iloc[0] = trajectory['dy'].iloc[0] = trajectory['dz'].iloc[0] = 0 
    trajectory['dvx'].iloc[0] = trajectory['dvy'].iloc[0] = trajectory['dvz'].iloc[0] = 0 
    return torch.tensor(trajectory.values, dtype=torch.float32), torch.tensor(maneuvers)

def postprocess_joint(joint_dist):
    """ casts joint distribution to a dataframe, and appends the predicted maneuver
    """
    joint_df = pd.DataFrame({
            'takeoff': joint_dist[:, 0],
            'turn': joint_dist[:, 1],
            'line': joint_dist[:, 2],
            'orbit': joint_dist[:, 3],
            'landing': joint_dist[:, 4]
        })
    joint_df['maneuver'] = joint_df.idxmax(axis="columns")
    return joint_df

def train(num_valid, max_steps, model, num_test=0, sampling_interval=30):
    
    # number of train examples to use per epoch
    for NUM_TRAIN in [5, 10, 100]:
        L.seed_everything(0)

        datamodule = FlightTrajectoryDataModule(NUM_TRAIN, num_valid, num_test)

        model = model(
            k_size=[3] * 300, 
            c_hidden=[36] * 300,
            d_size=[5] * 300,
            block_type="ResNetBlock"
        )

        logger = TensorBoardLogger(EXPERIMENT_PATH, name="resnet" + '-train-' + str(NUM_TRAIN))
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = Trainer(logger=logger, callbacks=[lr_monitor], log_every_n_steps=10, max_steps=MAX_STEPS)
        trainer.fit(model, dataset)