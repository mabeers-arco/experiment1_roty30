""" 
Run this app with `python app.py` and visit http://127.0.0.1:8050/ in your web browser.
To do: 
3) Bigger images/figures
5) lighting and materials
"""
from dash import Dash, html, dcc, Input, Output
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from base64 import b64encode
import pickle

# dataset_path = './data/trial_dataset.pickle'
# dataset_path = './data/experiment1.pickle'
dataset_path = "./assets/exp1ty30.pickle"
with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

# trial 0 
xyz = dataset['trial0']['left']
xyz2 = dataset['trial0']['right']
refaced = dataset['trial0']['ijk']
trial_names = list(dataset.keys())
possible_responses = ['Left Looks Better', 'Undecided', "Right Looks Better"]
responses = {trial:'Undecided' for trial in dataset}

app = Dash(__name__)
server = app.server

@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value"))
def update_lower_graph(trial):
    xyz = dataset[trial]['left']
    xyz2 = dataset[trial]['right']
    refaced = dataset[trial]['ijk']

    fig = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.05, specs= [ [{"type":'scene'}, {"type":'scene'}]])

    left = go.Mesh3d(x=xyz[:, 0], 
        y=xyz[:, 1], 
        z=xyz[:, 2], 
        color='lightgrey', 
        i=refaced[:, 0], 
        j=refaced[:, 1],
        k=refaced[:, 2],
        flatshading=True)

    right = go.Mesh3d(x=xyz2[:, 0], 
        y=xyz2[:, 1], 
        z=xyz2[:, 2], 
        color='lightgrey', 
        i=refaced[:, 0], 
        j=refaced[:, 1],
        k=refaced[:, 2], 
        flatshading=True)

    fig['layout']['scene'].update(aspectmode="data")
    fig['layout']['scene']['camera']['up'] = {"x":0, "y":1, "z":0}
    fig['layout']['scene']['camera']['eye'] = {"x":0, "y":0, "z":-30}
    fig['layout']['scene']['camera']['center'] = {"x":0, "y":0, "z":0}
    fig['layout']['scene']['camera']['projection'].update(type="orthographic")
    fig['layout']['scene']['xaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig['layout']['scene']['yaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig['layout']['scene']['zaxis'].update(showgrid=False, zeroline=False, visible=False)

    fig['layout']['scene2'].update(aspectmode="data")
    fig['layout']['scene2']['camera']['up'] = {"x":0, "y":1, "z":0}
    fig['layout']['scene2']['camera']['eye'] = {"x":0, "y":0, "z":-30}
    fig['layout']['scene2']['camera']['center'] = {"x":0, "y":0, "z":0}
    fig['layout']['scene2']['camera']['projection'].update(type="orthographic")
    fig['layout']['scene2']['xaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig['layout']['scene2']['yaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig['layout']['scene2']['zaxis'].update(showgrid=False, zeroline=False, visible=False)

    fig.add_trace(left, row=1, col=1)
    fig.add_trace(right, row=1, col=2)
    fig.update_layout(height=600, width = 1500)
    return fig



@app.callback(
    Output("repeater", "children"),
    Input("radio", "value"),
    Input("dropdown", "value"))
def update_storage(response, trial):
    """
    Whenever we make a choice regarding which reconstruction looks better, we want to save this. 
    """
    responses[trial] = response

@app.callback(
    Output("radio", "value"),
    Input("dropdown", "value"))
def update_radio(trial):
    """
    When we update the dropdown box to go to a different trial, we want to change the radio button
    selection to whatever it used to be. 
    """
    return responses[trial]


@app.callback(
    Output("download_txt", "data"),
    Input("download_button", "n_clicks"))
def download_func(n_clicks):
    if n_clicks:
        df = pd.DataFrame(np.zeros((len(responses), 4)), columns = ["trial", "left_source", "right_source", "selection"])
        df.loc[:, "selection"] = [responses[trial] for trial in dataset]
        df.loc[:, "right_source"] = [dataset[trial]['right_source'] for trial in dataset]
        df.loc[:, "left_source"] = [dataset[trial]['left_source'] for trial in dataset]
        df.loc[:, "trial"] = list(dataset.keys())
        return dcc.send_data_frame(df.to_csv, "output.csv")


@app.callback(
    Output(component_id="im", component_property="src"),
    Input("dropdown", "value"))
def update_upper_graph(trial):
    xyz = dataset[trial]['left']
    xyz2 = dataset[trial]['right']
    refaced = dataset[trial]['ijk']

    fig_static = go.Figure(go.Mesh3d(x=xyz[:, 0], 
        y=xyz[:, 1], 
        z=xyz[:, 2], 
        color='lightgrey', 
        i=refaced[:, 0], 
        j=refaced[:, 1],
        k=refaced[:, 2],
        flatshading=True))


    fig_static['layout']['scene'].update(aspectmode="data")
    fig_static['layout']['scene']['camera']['up'] = {"x":0, "y":1, "z":0}
    fig_static['layout']['scene']['camera']['eye'] = {"x":0, "y":0, "z":-50}
    fig_static['layout']['scene']['camera']['center'] = {"x":0, "y":0, "z":0}
    fig_static['layout']['scene']['camera']['projection'].update(type="orthographic")
    fig_static['layout']['scene']['xaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig_static['layout']['scene']['yaxis'].update(showgrid=False, zeroline=False, visible=False)
    fig_static['layout']['scene']['zaxis'].update(showgrid=False, zeroline=False, visible=False)

    img_bytes = fig_static.to_image(format="png")
    encoding = b64encode(img_bytes).decode()
    img_b64 = "data:image/png;base64," + encoding
    return img_b64
   


app.layout = html.Div(children=[
    html.H1(children='Interface Demo'),

    html.Div(children='''
        This is what an interface might look like with Plotly
    '''),
    dcc.Dropdown(
        id = 'dropdown',
        options=trial_names,
        value='trial0',
        clearable=False
    ),

    dcc.RadioItems(
        id = "radio",
        options=possible_responses,
        value=responses['trial0'], 
        inline=True),
    html.Div(children="Undecided", id="repeater"),
    html.Img(src=update_upper_graph('trial0'), style={"height":"600px"}, id='im'),

    dcc.Graph(
        id='graph'
    ),
    html.Div([html.Button("Download Results", id='download_button'),
        dcc.Download(id="download_txt")])
    

])




if __name__ == '__main__':
    app.run_server(debug=True)
