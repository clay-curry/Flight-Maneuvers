import os
from src import EXPERIMENT_PATH
from src.data_module import FlightTrajectoryDataModule, MANEUVERS
from src.model.resnet import ResNet

import pandas as pd
import plotly.express as px
from scipy.ndimage import rotate
import plotly.graph_objects as go
from flask import Dash, dcc, html
from dash.dependencies import Input, Output

dataloader = FlightTrajectoryDataModule(num_test=10).test_dataloader()

demo_data = {dataloader.dataset.files[i][26:]: demo for i, demo in enumerate(dataloader.dataset)}

model_names = [name for name in os.listdir(EXPERIMENT_PATH) if os.path.isdir(os.path.join(EXPERIMENT_PATH, name))]

app = Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'cmap': {
        'takeoff': "red",
        'turn': "green",
        'line': "blue",
        'orbit': "goldenrod",
        'landing': "magenta"
    },
}


app.layout = html.Div(
    # main container
    html.Div(
        # body
        html.Div(
            [
                # left column
                html.Div([
                    html.H1(children='Project Demo'),
                    
                    html.Div(
                        [
                            html.H3(children='Simulation'),
                            dcc.Dropdown(
                                    list(demo_data.keys()),
                                    list(demo_data.keys())[0],
                                    id='scenario-name',
                                    style={'width': '25vw'}
                                ),
                        ]
                    ),

                    html.Div(
                        [
                            html.H3(children='Pretrained Model'),
                            dcc.Dropdown(
                                model_names,
                                model_names[0],
                                id='network-selection', 
                                
                            )
                        ]
                    ),

                    html.Div(
                        [
                            html.H3(children='Maneuver'),
                            dcc.RadioItems(
                                ['All'] + [m.capitalize() for m in MANEUVERS],
                                'All',
                                id='maneuver-name', inline=True, style={'padding': '10px'}
                            )
                        ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-evenly'}
                    ),

                    html.Div(
                        [
                            html.H3(children='Reference Frame'),
                            
                            html.Div(
                                [
                                    html.Div(children='θ   = '),
                                    html.Div(id='theta-output-container'),

                                    html.Div(   
                                    dcc.Slider(-3.14, 3.14,
                                        value=0,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        id='theta-slider',
                                        
                                    ), style={'width': '70%'})

                                ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-evenly'}
                            )
                        ]
                    )

                    ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-evenly', 'gap': '10px', 'border': '1px solid gray', 'padding': '40px', 'border-radius': '10px'}
                ),
                
                # right column
                html.Div(
                    [
                        dcc.Graph(id='true-maneuvers'),
                        dcc.Graph(id='pred-maneuvers'),
                    ],
                    style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'space-evenly'}
                )
            ],
            style={'display': 'flex', 'align-items': 'center', 'gap': '5px', 'justify-content': 'space-evenly', 'height': '80vh', 'flex-direction': 'row', 'flex-wrap': 'wrap'}
        ), style={'max-width': '1250px', 'border': '1px solid gray', 'padding': '40px', 'border-radius': '50px'}
    ), style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'flex-direction': 'column' }
)
    
    
@app.callback(
    Output('theta-output-container', 'children'),
    Input('theta-slider', 'value'))
def update_theta(value):
    return '{}'.format(value)

@app.callback(
    Output('true-maneuvers', 'figure'),
    Input('scenario-name', 'value'),
    Input('maneuver-name', 'value'),
    Input('theta-slider', 'value')
)

def update_true_trajectory(value, selected_maneuver, theta):
    
    scenario = demo_data[value].copy()
    bound = max(max(scenario['x']) - min(scenario['x']), max(scenario['y']) - min(scenario['y'])) * 1.2
    scenario['x'] -= (max(scenario['x']) + min(scenario['x'])) / 2
    scenario['y'] -= (max(scenario['y']) + min(scenario['y'])) / 2

    if selected_maneuver == "All":
        fig = px.scatter(scenario, x="x", y="y", color="maneuver", color_discrete_map=styles['cmap'], hover_data=['t', 'maneuver'])        
        fig.update_layout(
            title="True Maneuvers",
            clickmode='event+select',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=10, b=10)
        )
        
    
    else:
        scenario[selected_maneuver] = 0
        scenario.loc[scenario['maneuver'] == selected_maneuver.lower(), selected_maneuver] = 1
        fig = px.scatter(scenario, x="x", y="y", color=selected_maneuver, range_color=[0,1],
                         title="True Maneuvers", color_continuous_scale=[[0, 'rgb(150, 150, 250)'], [1, 'rgb(250, 0, 0)']])
        fig.update_layout(
            legend=dict(yanchor="bottom", y=0.30, xanchor="left", x=0),
            margin=dict(l=50, r=50, t=30, b=10)
        )
        
    fig.update_xaxes(range=[-bound / 2, bound / 2])
    fig.update_yaxes(range=[-bound / 2, bound / 2])
    return fig
    

@app.callback(
    Output('pred-maneuvers', 'figure'),
    Input('scenario-name', 'value'),
    Input('maneuver-name', 'value'),
    Input('network-selection', 'value'),
    Input('theta-slider', 'value')
)
def update_pred_trajectory(scenario, selected_maneuver, network_selection, theta):
    
    # load scenario and adjust reference frame
    scenario = demo_data[scenario].copy()
    bound = max(max(scenario['x']) - min(scenario['x']), max(scenario['y']) - min(scenario['y'])) * 1.2
    scenario['x'] -= (max(scenario['x']) + min(scenario['x'])) / 2
    scenario['y'] -= (max(scenario['y']) + min(scenario['y'])) / 2

    if 'resnet' in network_selection:
        print('loading resnet (', network_selection, ')')
        weight_dir = os.path.join(EXPERIMENT_PATH, network_selection, 'version_0', 'checkpoints')
        model = ResNet.load_from_checkpoint(os.path.join(weight_dir, os.listdir(weight_dir)[0]))
        predicted_maneuvers = model.predict(scenario)
    
    
    if selected_maneuver == "All":
        scenario['maneuver'] = predicted_maneuvers['maneuver']
        
        fig = px.scatter(scenario, x="x", y="y", color="maneuver", color_discrete_map=styles['cmap'], hover_data=['t', 'maneuver'])
        
        fig.update_layout(
            title="Predicted Maneuvers",
            clickmode='event+select',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=10, b=10)
        )
        
    
    else:
        scenario[selected_maneuver] = predicted_maneuvers[selected_maneuver.lower()]
        
        fig = px.scatter(scenario, x="x", y="y", color=selected_maneuver, range_color=[0,1],
                         title="Predicted Maneuvers", color_continuous_scale=[[0, 'rgb(150, 150, 250)'], [1, 'rgb(250, 0, 0)']])
        
        fig.update_layout(
            legend=dict(yanchor="bottom", y=0.30, xanchor="left", x=0),
            margin=dict(l=50, r=50, t=30, b=10)
        )
        
    fig.update_xaxes(range=[-bound / 2, bound / 2])
    fig.update_yaxes(range=[-bound / 2, bound / 2])
    return fig


if __name__ == '__main__':
        app.run_server(debug=True)
