 # %%
import os
import numpy as np
import pandas as pd
import open3d as o3d 

import plotly.graph_objs as go

 # %%
ori_pc_path = './test_sample/original_bag.ply'
adv_pc_path = './testa_sample/attack_sample_bag.ply'

pc = o3d.io.read_point_cloud(ori_pc_path)
adv_pc = o3d.io.read_point_cloud(adv_pc_path)

np_pc = np.asarray(pc.points)
adv_np = np.asarray(adv_pc.points)

x_pc = np_pc[:,0]
y_pc = np_pc[:,1]
z_pc = np_pc[:,2]

adv_x_pc = adv_np[:,0]
adv_y_pc = adv_np[:,1]
adv_z_pc = adv_np[:,2]

ori_scatter = go.Scatter3d(
    x=x_pc,  # <-- Put your data instead
    y=y_pc,  # <-- Put your data instead
    z=z_pc,  # <-- Put your data instead
    mode='markers',
    marker={
        'size': 4,
        'opacity': 1,
        'color': 'black'
    },
)

adv_scatter = go.Scatter3d(
    x=adv_x_pc,  # <-- Put your data instead
    y=adv_y_pc,  # <-- Put your data instead
    z=adv_z_pc,  # <-- Put your data instead
    mode='markers',
    marker={
        'size': 4,
        'opacity': 1,
        'color':'red'
    },
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    paper_bgcolor='rgba(255,255,255,255)',
    plot_bgcolor='rgba(0,0,0,0)',
    width = 1000,
    height = 900,
    scene = dict(
    xaxis = dict(showgrid=False, showspikes=False,showline=False, 
                 zeroline=False, showbackground=False, showticklabels=False, title=''),
    yaxis = dict(showgrid=False, showspikes=False, showline=False, 
                 zeroline=False,  showbackground=False, showticklabels=False, title=''),
    zaxis = dict(showgrid=False, showspikes=False,showline=False, 
                 zeroline=False,  showbackground=False, showticklabels=False, title=''),
    ),
    showlegend=False
)
ori_figure = go.Figure(data=[ori_scatter], layout=layout)
adv_figure = go.Figure(data=[adv_scatter,ori_scatter], layout=layout)

ori_figure.show()
adv_figure.show()



# %%
