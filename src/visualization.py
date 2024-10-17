import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from matplotlib import cm

colors = ['blue', 'orange', 'green', 'purple', 'orange']
shapes = ['solid', 'solid', 'dashdot', 'longdash', 'longdashdot']
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 32

def name_to_rgb(name):
    rgb = mcolors.to_rgb(name)  # This will get the RGB values in a range of 0-1
    r, g, b = [int(x*255) for x in rgb]  # Scaling RGB values to 0-255
    return r, g, b


colors =  [
    'rgb(230, 159, 0)',   # Orange
    'rgb(0, 158, 115)',   # Bluish Green
    'rgb(0, 114, 178)',   # Blue
    'rgb(204, 121, 167)'  # Reddish Purple
]
rgb_colors =  [
    'rgb(230, 159, 0)',   # Orange
    'rgb(0, 158, 115)',   # Bluish Green
    'rgb(0, 114, 178)',   # Blue
    'rgb(204, 121, 167)'  # Reddish Purple
]

# Convert to matplotlib format
mpl_colors = [tuple(int(c) / 255 for c in color[4:-1].split(',')) for color in rgb_colors]

def replace_keys_with_names(result_dict, lookup_dict):
    """
    replace keys with names

    Parameters
    ----------
    result_dict : dict
        result dictionary
    lookup_dict : dict
        dictionary to look up keys and get replacement

    Returns
    -------
    dict
        result dictionary with replaced keys
    """
    return {lookup_dict[key]: value for key, value in result_dict.items() if key in lookup_dict}


def load_bounds(res, res_seed, step_size):
    """
    Load data from results

    Parameters
    ----------
    res : dict
        result dictionary for single run
    res_seed : dict
        result dictionary for all different seeds
    step_size : np.float
        value of step size

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds, lower_bounds_seeds, upper_bounds_seeds, res_time)
    """
    lower_bounds_seeds = {}
    upper_bounds_seeds = {}
    for strat, res_strat in res_seed.items():
        try: 
            lower_bounds_seeds.update({strat: np.array(res_strat["Grad_Min_Exact"]).squeeze()})
            upper_bounds_seeds.update({strat: np.array(res_strat["Grad_Max_Exact"]).squeeze()})

            if strat == "Adaptive":
                try:
                    lower_bounds_seeds["Adaptive"] = lower_bounds_seeds["Adaptive"][:, step_size, :].squeeze()
                    upper_bounds_seeds["Adaptive"] = upper_bounds_seeds["Adaptive"][:, step_size, :].squeeze()
                except IndexError:
                    lower_bounds_seeds["Adaptive"] = lower_bounds_seeds["Adaptive"].squeeze()
                    upper_bounds_seeds["Adaptive"] = upper_bounds_seeds["Adaptive"].squeeze()
        except TypeError:
            print(f"{strat} not available")
    
    res_time = {}
    try:
        for strat, res_strat in res_seed.items():
            try: 
                res_time.update({strat: np.array(res_strat["Time"]).squeeze()})
            
                if strat == "Adaptive":
                    try:
                        res_time["Adaptive"] = res_time["Adaptive"][:, step_size, :].squeeze()
                    except IndexError:
                        res_time["Adaptive"] = res_time["Adaptive"].squeeze()
            except TypeError:
                print(f"{strat} not available")
    except KeyError:
        print(f"Time not available")

    lower_bounds = {}
    upper_bounds = {}
    for strat, res_strat in res.items():
        try:
            lower_bounds.update({strat: np.array(res_strat["Grad_Min_Exact"]).squeeze()})
            upper_bounds.update({strat: np.array(res_strat["Grad_Max_Exact"]).squeeze()})
            if strat == "Adaptive":
                try:
                    lower_bounds["Adaptive"] = lower_bounds["Adaptive"][step_size]#.squeeze()
                    upper_bounds["Adaptive"] = upper_bounds["Adaptive"][step_size]#.squeeze()
                except IndexError:
                    print(f"IndexError: {strat}")
        except TypeError:
            print(f"{strat} not available")
    return lower_bounds, upper_bounds, lower_bounds_seeds, upper_bounds_seeds, res_time


def load_results(output_path, experiment_name, strategy_type, seed, lam_c, T, T_exploration, sample_sigma, iteration, lam_first, do_update):
    """
    Load results from files

    Parameters
    ----------
    output_path : string
        output path
    experiment_name : string
        name of experiment
    strategy_type : string
        name of strategy for which to load results
    seed : int
        seed for which to load results
    lam_c : np.float
        hyperparameter for the regularization of feasibility constraint
    T : int
        length of experiment
    T_exploration : int
        length of exploration phase of experiment
    sample_sigma : np.float
        sigma value for instrument sampling
    iteration : int
        iteration for single results
    lam_first : np.float
        hyperparameter for function regularization
    do_update : np.bool 
        whether to update sigma

    Returns
    -------
    tuple of dict   
        (res, res_seed)
    """
    
    res = {}
    res_seed = {}
    for i in range(len(strategy_type)):
        
        if do_update==None:
            folder_name = f"strategy_type-{strategy_type[i]}_lam_c-{lam_c}_lam_first-{lam_first}_seed-{seed}_T-{T}_T_exploration-{T_exploration}_sample_sigma-{sample_sigma}"
        else:
            folder_name = f"strategy_type-{strategy_type[i]}_lam_c-{lam_c}_lam_first-{lam_first}_seed-{seed}_T-{T}_T_exploration-{T_exploration}_sample_sigma-{sample_sigma}_do_sigma_update-{do_update}"
        file_name = os.path.join(output_path, experiment_name, folder_name, f"results_{iteration}.npy")
        res_name = os.path.join(output_path, experiment_name, folder_name, f"results_seed.npy")
        try:
            res_single = np.load(file_name, allow_pickle=True).item()
            res_single_seed = np.load(res_name, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"File--- {strategy_type[i]} ---not found for {lam_c}, {lam_first}, {do_update}")
            res_single = None
            res_single_seed = None
        res.update({strategy_type[i]: res_single})
        res_seed.update({strategy_type[i]: res_single_seed})


    return res, res_seed


def update_layout(fig, do_markers=True):
    """update layout for the paper

    Parameters
    ----------
    fig : plotly figure

    Returns
    -------
    fig : plotly figure
        input figure with updated layout

    """

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="serif", size=32),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),  # gridcolor="grey"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black")  # , gridcolor="grey")
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
    if do_markers:
        fig.update_traces(marker_line_width=2, marker_size=5, line_width=4)
    fig.update_layout(layout)

    return fig


def visualize_grad_confidence_mse(lower_bounds, grad_true, len_=None, range_=None):
    fig = go.Figure()

    length_ = 0
    for i, key in enumerate(lower_bounds.keys()):
        if len_ == None:
            length = len(lower_bounds[key][0])
            if length > length_:
                length_ = length
        else:
            length_ = len_

    conf_u = 90.0
    conf_l = 10.0
    for i, key in enumerate(lower_bounds.keys()):
        print(key)
        def pad_array(array, length_):
            if np.isscalar(array):
                array = np.array([array])
            if len(array) < length_:
                array = np.concatenate((np.full(length_ - len(array), np.nan), array))
            return array
        if key == "Alternating":
            mean_lower = np.repeat(np.mean((lower_bounds[key] - np.mean(grad_true))**2, axis=0), 2)
            per_lower_l = np.repeat(np.percentile((lower_bounds[key] - np.mean(grad_true))**2, conf_l, axis=0), 2)
            per_lower_u = np.repeat(np.percentile((lower_bounds[key] - np.mean(grad_true))**2, conf_u, axis=0), 2)
        else:
            # Compute the column-wise mean, lower and upper confidence bounds
            mean_lower = pad_array(np.mean((lower_bounds[key] - np.mean(grad_true))**2, axis=0), length_)
            per_lower_l = pad_array(np.percentile((lower_bounds[key] - np.mean(grad_true))**2, conf_l, axis=0), length_)
            per_lower_u = pad_array(np.percentile((lower_bounds[key] - np.mean(grad_true))**2, conf_u, axis=0), length_)
        
        try:
            if len_ == None:
                len_ = len(lower_bounds[key])
            # Change scatter to lines and use different line shapes for lower and upper bounds
            fig.add_trace(go.Scatter(x=list(range(len_)), y=mean_lower, mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[0]), name=f"{key}", showlegend=True))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_lower_l, mode='lines', line=dict(color=colors[i % len(colors)], width=0), name=f"{key}", showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_lower_u, mode='lines', line=dict(color=colors[i % len(colors)], width=0), 
                                    name=f"{key}", showlegend=False, fill="tonexty", 
                                    fillcolor=f'rgba({int(rgb_colors[i % len(rgb_colors)].split("(")[1].split(",")[0])},{int(rgb_colors[i % len(rgb_colors)].split(",")[1])},{int(rgb_colors[i % len(rgb_colors)].split(",")[2].split(")")[0])},0.1)'))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=np.zeros(len_), mode='lines', line=dict(color='black'), name='True Gradient', showlegend=False))
        except ValueError:
            print(f"ValueError: {key}")
        if range_ is not None:
            # Calculate the mean of grad_true
            mean_grad_true = np.mean(grad_true)
            fig.update_yaxes(range=[mean_grad_true - range_, mean_grad_true + range_])
    
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
)
    return fig


def visualize_grad_confidence_final_mse_overall(res_overall, strategy_name, len_=None, range_=None):
    """
    Visualize gradient upper and lower bounds for one strategy and all run hyperparameters

    Parameters
    ----------
    res_overall : dict
        result dictionary with hyperparameters as keys
    strategy_name : string
        name of strategy
    len_ : int, optional
        length of experiment, by default None
    range_ : int, optional
        range of y-axis, by default None

    Returns
    -------
    plotly figure
        Figure of hyperparameters vs. bounds
    """
    fig = go.Figure()
    
    key = strategy_name
    for i, key_overall in enumerate(res_overall.keys()):
        try:
            j = 2
            lower_bounds = res_overall[key_overall]["lower_bounds_seeds"]
            upper_bounds = res_overall[key_overall]["upper_bounds_seeds"]
            
            grad_true = res_overall[key_overall]["grad_true"]
            lower = lower_bounds[key][:, -1]
            upper = upper_bounds[key][:, -1]

            if len_ == None:
                len_ = len(lower_bounds[key])
        
            # Change scatter to lines and use different line shapes for lower and upper bounds
            fig.add_trace(go.Box(y=lower, name=f"{key_overall}", marker=dict(color=colors[j]), showlegend=False))
            fig.add_trace(go.Box(y=upper, name=f"{key_overall}", marker=dict(color=colors[j]), showlegend=False))
        
            if range_ is not None:
                # Calculate the mean of grad_true
                mean_grad_true = np.mean(grad_true)
                fig.update_yaxes(range=[mean_grad_true-range_, mean_grad_true+range_])

            
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=mean_grad_true,
                x1=i+0.5,
                y1=mean_grad_true,
                line=dict(
                    color="Black",
                    width=3,
                ),
            )
        except KeyError:
            print(f"KeyError: {key_overall}")
    return fig


def visualize_grad_confidence(lower_bounds, upper_bounds, grad_true, len_=None, range_=None):
    """
    Visualize gradient upper and lower bounds for multiple strategies over all experiments until len_

    Parameters
    ----------
    lower_bounds : np.ndarray
        matrix of lower boundsd
    upper_bounds : np.ndarray
        matrix of upper bounds
    grad_true : np.ndarray
        true gradient of the function
    len_ : int, optional
        length of experimental rounds, by default None
    range_ : int, optional
        range of y-axis, by default None

    Returns
    -------
    plotly figure  
        plotly figure of gradient bounds over all experiments
    """
    fig = go.Figure()

    length_ = 0
    for i, key in enumerate(lower_bounds.keys()):
        if len_ == None:
            length = len(lower_bounds[key][0])
            if length > length_:
                length_ = length
        else:
            length_ = len_

    conf_u = 90.0
    conf_l = 10.0
    for i, key in enumerate(lower_bounds.keys()):
        print(key)
        def pad_array(array, length_):
            if np.isscalar(array):
                array = np.array([array])
            if len(array) < length_:
                array = np.concatenate((np.full(length_ - len(array), np.nan), array))
            return array
        if key == "Alternating":
            mean_lower = np.repeat(np.mean(lower_bounds[key], axis=0), 2)
            per_lower_l = np.repeat(np.percentile(lower_bounds[key], conf_l, axis=0), 2)
            per_lower_u = np.repeat(np.percentile(lower_bounds[key], conf_u, axis=0), 2)

            #upper_bounds[key] = pad_array(upper_bounds[key], length_)
            mean_upper = np.repeat(np.mean(upper_bounds[key], axis=0), 2)
            per_upper_l = np.repeat(np.percentile(upper_bounds[key], conf_l, axis=0), 2)
            per_upper_u = np.repeat(np.percentile(upper_bounds[key], conf_u, axis=0), 2)
        else:
            # Compute the column-wise mean, lower and upper confidence bounds
            mean_lower = pad_array(np.mean(lower_bounds[key], axis=0), length_)
            per_lower_l = pad_array(np.percentile(lower_bounds[key], conf_l, axis=0), length_)
            per_lower_u = pad_array(np.percentile(lower_bounds[key], conf_u, axis=0), length_)

            #upper_bounds[key] = pad_array(upper_bounds[key], length_)
            mean_upper = pad_array(np.mean(upper_bounds[key], axis=0), length_)
            per_upper_l = pad_array(np.percentile(upper_bounds[key], conf_l, axis=0), length_)
            per_upper_u = pad_array(np.percentile(upper_bounds[key], conf_u, axis=0), length_)
        

        try:
            if len_ == None:
                len_ = len(lower_bounds[key])
            # Change scatter to lines and use different line shapes for lower and upper bounds
            fig.add_trace(go.Scatter(x=list(range(len_)), y=mean_lower, mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[0]), name=f"{key}", showlegend=True))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_lower_l, mode='lines', line=dict(color=colors[i % len(colors)], width=0), name=f"{key}", showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_lower_u, mode='lines', line=dict(color=colors[i % len(colors)], width=0), 
                                    name=f"{key}", showlegend=False, fill="tonexty", 
                                    fillcolor=f'rgba({int(rgb_colors[i % len(rgb_colors)].split("(")[1].split(",")[0])},{int(rgb_colors[i % len(rgb_colors)].split(",")[1])},{int(rgb_colors[i % len(rgb_colors)].split(",")[2].split(")")[0])},0.1)'))
            
            fig.add_trace(go.Scatter(x=list(range(len_)), y=mean_upper, mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[1]), name=f"{key}", showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_upper_l, mode='lines', line=dict(color=colors[i % len(colors)], width=0), name=f"{key}", showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=per_upper_u, mode='lines', line=dict(color=colors[i % len(colors)], width=0), 
                                    name=f"{key}", showlegend=False, fill="tonexty", 
                                    fillcolor=f'rgba({int(rgb_colors[i % len(rgb_colors)].split("(")[1].split(",")[0])},{int(rgb_colors[i % len(rgb_colors)].split(",")[1])},{int(rgb_colors[i % len(rgb_colors)].split(",")[2].split(")")[0])},0.1)'))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=grad_true[:len_], mode='lines', line=dict(color='black'), name='True Gradient', showlegend=False))
        except ValueError:
            print(f"ValueError: {key}")
        if range_ is not None:
            # Calculate the mean of grad_true
            mean_grad_true = np.mean(grad_true)
            fig.update_yaxes(range=[mean_grad_true - range_, mean_grad_true + range_])
    
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
)
    return fig


def visualize_grad_confidence_final(lower_bounds, upper_bounds, grad_true, len_=None, range_=None):
    """
    Visualize gradient upper and lower bounds for multiple strategies over final experiment

    Parameters
    ----------
    lower_bounds : np.ndarray
        lower bounds of the gradient
    upper_bounds : np.ndarray
        upper bounds of the gradient
    grad_true : np.ndarray
        true gradient of the function
    len_ : int, optional
        length of visualization, number of experiments, by default None
    range_ : int, optional
        range of y-axis, by default None

    Returns
    -------
    plotly figure
        plotly figure of gradient bounds over final experiment and over all seeds
    """
    fig = go.Figure()

    for i, key in enumerate(lower_bounds.keys()):
        lower = lower_bounds[key][:, -1]
        upper = upper_bounds[key][:, -1]

        if len_ == None:
            len_ = len(lower_bounds[key])
        
        # Change scatter to lines and use different line shapes for lower and upper bounds
        fig.add_trace(go.Box(y=lower, name=f"{key}", marker=dict(color=colors[i % len(colors)]), showlegend=False))
        fig.add_trace(go.Box(y=upper, name=f"{key}", marker=dict(color=colors[i % len(colors)]), showlegend=False))
        fig.update_layout(
            boxmode='overlay'  # group together boxes of the different traces for each value of x
        )

        if range_ is not None:
            # Calculate the mean of grad_true
            mean_grad_true = np.mean(grad_true)
            fig.update_yaxes(range=[mean_grad_true - range_, mean_grad_true + range_])

        mean_grad_true = np.mean(grad_true)

        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=mean_grad_true,
            x1=i+0.5,
            y1=mean_grad_true,
            line=dict(
                color="Black",
                width=3,
            ),
        )
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0,
            x1=i+0.5,
            y1=0,
            line=dict(
                color="Black",
                width=3,
                dash="dot",
            ),
        )
    return fig


def visualize_grad_confidence_final_mse(lower_bounds, upper_bounds, grad_true, len_=None, range_=None):
    """
    Visualize gradient upper and lower bound errors to true gradient for multiple strategies over final experiment

    Parameters
    ----------
    lower_bounds : np.ndarray
        lower bounds of the gradient
    upper_bounds : np.ndarray
        upper bounds of the gradient
    grad_true : np.ndarray
        true gradient of the function
    range_ : int, optional
        range of y-axis, by default None

    Returns
    -------
    plotly figure
        plotly figure of gradient errors of final experiment and over all seeds
    """
    fig = go.Figure()
    
    for i, key in enumerate(lower_bounds.keys()):
            
        lower = lower_bounds[key][:, -1]
        upper = upper_bounds[key][:, -1]

        if len_ == None:
            len_ = len(lower_bounds[key])
        # Change scatter to lines and use different line shapes for lower and upper bounds
        
        fig.add_trace(go.Box(y=(lower - np.mean(grad_true))**2, name=f"{key}", marker=dict(color=colors[i % len(colors)]), showlegend=False))
        fig.add_trace(go.Box(y=(upper - np.mean(grad_true))**2, name=f"{key}", marker=dict(color=colors[i % len(colors)]), showlegend=False))
        fig.update_layout(
            boxmode='group'  # group together boxes of the different traces for each value of x
        )

        if range_ is not None:
            # Calculate the mean of grad_true
            mean_grad_true = np.mean(grad_true)
            fig.update_yaxes(range=[mean_grad_true - range_, mean_grad_true + range_])

        
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0,
            x1=i+0.5,
            y1=0,
            line=dict(
                color="Black",
                width=3,
                #dash="dot",
            ),
        )
    return fig


def visualize_time(lower_bounds, len_=None, range_=None):
    """
    Visualize the time taken for multiple strategies over all experiments

    Parameters
    ----------
    lower_bounds : np.ndarray
        time values over different strategies
    len_ : int, optional
        length of visualization, number of experiments, by default None
    range_ : int, optional
        range of y-axis, by default None

    Returns
    -------
    plotly figure 
        figure of time taken over all experiments
    """
    fig = go.Figure()

    length_ = 0
    for i, key in enumerate(lower_bounds.keys()):
        if len_ == None:
            length = len(lower_bounds[key][0])
            if length > length_:
                length_ = length
        else:
            length_ = len_

    
    for i, key in enumerate(lower_bounds.keys()):
        print(key)
        def pad_array(array, length_):
            if np.isscalar(array):
                array = np.array([array])
            if len(array) < length_:
                array = np.concatenate((np.full(length_ - len(array), np.nan), array))
            return array
        if key == "Alternating":
            mean_lower = np.repeat(np.mean(lower_bounds[key], axis=0), 2)
            per_lower_l = np.repeat(np.percentile(lower_bounds[key], 25.0, axis=0), 2)
            per_lower_u = np.repeat(np.percentile(lower_bounds[key], 75.0, axis=0), 2)

        else:
            # Compute the column-wise mean, lower and upper confidence bounds
            mean_lower = pad_array(np.mean(lower_bounds[key], axis=0), length_)
            per_lower_l = pad_array(np.percentile(lower_bounds[key], 25.0, axis=0), length_)
            per_lower_u = pad_array(np.percentile(lower_bounds[key], 75.0, axis=0), length_)

        try:
            #if len_ == None:
            #    len_ = len(lower_bounds[key])
            # Change scatter to lines and use different line shapes for lower and upper bounds
            fig.add_trace(go.Scatter(x=list(range(length_)), y=mean_lower, mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[0]), name=f"{key}", showlegend=True))
            fig.add_trace(go.Scatter(x=list(range(length_)), y=per_lower_l, mode='lines', line=dict(color=colors[i % len(colors)], width=0), name=f"{key}", showlegend=False))
            fig.add_trace(go.Scatter(x=list(range(length_)), y=per_lower_u, mode='lines', line=dict(color=colors[i % len(colors)], width=0), 
                                    name=f"{key}", showlegend=False, fill="tonexty", 
                                    fillcolor=f'rgba({int(rgb_colors[i % len(rgb_colors)].split("(")[1].split(",")[0])},{int(rgb_colors[i % len(rgb_colors)].split(",")[1])},{int(rgb_colors[i % len(rgb_colors)].split(",")[2].split(")")[0])},0.1)'))
            
        except ValueError:
            print(f"ValueError: {key}")
        
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
)
    return fig


def visualize_adaptive_update(res_seed, T_exploration, iteration, step_size_list):
    """
    Visualize adaptive update of gamma, mean and sigma

    Parameters
    ----------
    res_seed : dict
        dictionary of results over different seeds
    T_exploration : int
        number of exploration steps
    iteration : int
        seed for which to take results
    step_size_list : list
        list of step sizes

    Returns
    -------
    (tuple of plotly figures)
        (figure of gamma, figure of mean, figure of sigma)
    """
    fig1 = go.Figure()
    for s_size in range(res_seed["Adaptive"]["Gamma"][iteration].shape[0]):
        gamma_data = res_seed["Adaptive"]["Gamma"][iteration][s_size, :, :].squeeze()
        for gamma_k in range(gamma_data.shape[1]):
            fig1.add_trace(go.Scatter(x=np.arange(1, T_exploration+1), y=gamma_data[:, gamma_k], mode='lines', name=f"Step Size {step_size_list[s_size]}", \
                                      marker=dict(color=colors[s_size % len(colors)])))
            
    
    fig2 = go.Figure()
    for s_size in range(res_seed["Adaptive"]["Mu"][iteration].shape[0]):
        mu_data = res_seed["Adaptive"]["Mu"][iteration][s_size, :, :].squeeze()
        for mu_l in range(mu_data.shape[1]):
            fig2.add_trace(go.Scatter(x=np.arange(1, T_exploration+1), y=mu_data[:, mu_l], mode='lines', name=f"Step Size {step_size_list[s_size]}", \
                                      marker=dict(color=colors[s_size % len(colors)])))
    
    fig3 = go.Figure()
    for s_size in range(res_seed["Adaptive"]["Sigma"][iteration].shape[0]):
        sigma_data = res_seed["Adaptive"]["Sigma"][iteration][s_size, :, :].squeeze()
        for sigma_l in range(sigma_data.shape[1]):
            fig3.add_trace(go.Scatter(x=np.arange(1, T_exploration+1), y=sigma_data[:, sigma_l], mode='lines', name=f"Step Size {step_size_list[s_size]}", \
                                      marker=dict(color=colors[s_size % len(colors)])))

    return fig1, fig2, fig3


def visualize_iteration_debug(res, iteration_list, lr, grad_true, comp = 0, strat = None, step_size = 1):
    """
    Visualization of the sampled data of the different strategies, only viable for 2d functions

    Parameters
    ----------
    res : dict
        result dictionary
    iteration_list : list
        number of which experiments to visualize
    lr : np.float
        learning rate
    n_exploitation : int
        number of exploitation samples
    grad_true : np.ndarray
        true gradient of function
    comp : int, optional
        component of gradient, by default 0
    figsize : tuple, optional
        size of figure, by default (40, 40)
    strat : string, optional
        strategy name, by default None
    step_size : int, optional
        learning rate step in result, by default 1
    
    Returns
    -------
    plotly figure
        figure of sampled data for different steps in the experiment
    """
    X1 = res["X1"]
    X2 = res["X2"]
    f_grid = res["f_grid"]
    xbar = res["xbar"].squeeze()
    
    fig, ax = plt.subplots(1, len(iteration_list), figsize=(len(iteration_list) * 5, 5))  # Create subplots in a horizontal way

    for i, iteration in enumerate(iteration_list):
        x_exp = res["X_Samples"][lr][iteration]
        
        grad = np.zeros((2,))
        grad_min_exact = np.zeros((2,))
        grad_max_exact = np.zeros((2,))
        grad[comp] = grad_true.squeeze()
        
        if strat == "Adaptive":
            grad_min_exact[comp] = res["Grad_Min_Exact"].squeeze()[step_size, iteration]
            grad_max_exact[comp] = res["Grad_Max_Exact"].squeeze()[step_size, iteration]
        else:
            grad_min_exact[comp] = res["Grad_Min_Exact"].squeeze()[iteration]
            grad_max_exact[comp] = res["Grad_Max_Exact"].squeeze()[iteration]

        grad_results = {"Min": grad_min_exact.squeeze(), "Max": grad_max_exact.squeeze(), "Act": grad.squeeze()}

        # Plot isolines using the function values
        contour = ax[i].contour(X1, X2, f_grid, levels=10, cmap='coolwarm')
        
        # Plot gradient as vector at the average x position, if provided
        if xbar is not None and grad_results is not None:
            for j, (grad_name, grad_value) in enumerate(grad_results.items()):
                ax[i].quiver(*xbar, *grad_value, color=mpl_colors[j], scale=1, scale_units='xy', angles='xy', width=0.005, label=f"{grad_name}: {np.round(grad_value[comp], 1)}")

        ax[i].scatter(*xbar, marker='x', color='black', label='xbar', s=100)
        ax[i].grid(color='lightgray', linestyle='-', linewidth=0.5)
        ax[i].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)  # Decrease font size and position legend below

        # Decrease the font size of the tick labels
        ax[i].tick_params(axis='both', which='major', labelsize='x-small')

        # Use a colormap to get a unique color for each iteration
        color = cm.viridis(i / len(iteration_list))
        scatter = ax[i].scatter(x_exp[:, 0], x_exp[:, 1], alpha=0.5, s=1, color="blue")

    # Add a colorbar for the scatter plot
    fig.colorbar(scatter, ax=ax, label='Y values')


    return fig


def visualize_iteration(res, iteration, n_exploitation, grad_true, comp = 0, figsize = (40, 40), strat = None, step_size = 1, title = None):
    """
    Visualize sampled data of one strategy for one experiment iteration

    Parameters
    ----------
    res : dict
        result dictionary
    iteration : int
        number of experiment
    n_exploitation : int
        number of exploitation samples
    grad_true : np.ndarray
        true gradient of function
    comp : int, optional
        number of component, by default 0
    figsize : tuple, optional
        size of figure, by default (40, 40)
    strat : string, optional
        strategy name for which the experiment should be visualized, by default None
    step_size : int, optional
        step size, only necessary for adaptive strategy, by default 1
    
    Returns
    -------
    matplotlib figure
        figure of sampled data for one experiment iteration
    """
    X1 = res["X1"]
    X2 = res["X2"]
    f_grid = res["f_grid"]
    xbar = res["xbar"].squeeze()
    
    if strat == "Adaptive":
        x_exp = res["X_Samples"][step_size][iteration]
        y_exp = res["Y_Samples"][step_size][iteration]
    else:
        x_exp = res["X_Samples"][ : (iteration + 1) * n_exploitation, :]
        y_exp = res["Y_Samples"][ : (iteration + 1) * n_exploitation]

    grad = np.zeros((2, ))
    grad_min_exact = np.zeros((2, ))
    grad_max_exact = np.zeros((2, ))
    grad[comp] = grad_true.squeeze()
    if strat == "Adaptive":
        grad_min_exact[comp] = res["Grad_Min_Exact"].squeeze()[step_size, iteration]
        grad_max_exact[comp] = res["Grad_Max_Exact"].squeeze()[step_size, iteration]
    else:
        grad_min_exact[comp] = res["Grad_Min_Exact"].squeeze()[iteration]
        grad_max_exact[comp] = res["Grad_Max_Exact"].squeeze()[iteration]
        

    grad_results = {"Minimum ": grad_min_exact.squeeze(), "Maximum ": grad_max_exact.squeeze(), "Actual ": grad.squeeze()}
    
    fig, ax = plt.subplots(figsize=figsize)

    # Plot isolines using the function values
    contour = ax.contour(X1, X2, f_grid, levels=10, cmap='coolwarm')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient as vector at the average x position, if provided
    i = 0
    if xbar is not None and grad_results is not None:
        for grad_name, grad_value in grad_results.items():
            ax.quiver(*xbar, *grad_value, color=mpl_colors[i], scale=1, scale_units='xy', angles='xy', width=0.005, label=f"{grad_name}: {np.round(grad_value[comp], 1)}")

            i += 1
    
    ax.scatter(*xbar, marker='x', color='black', label='xbar', s=100)
    # Add grid lines
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('X0 - Component')
    ax.set_ylabel('X1 - Component')
    ax.legend()

    scatter = ax.scatter(x_exp[:, 0], x_exp[:, 1], c=y_exp, cmap='coolwarm', alpha=0.5, s=1)
    
    # Add a colorbar for the scatter plot
    fig.colorbar(scatter, ax=ax, label='Y values')

    return fig


def plot_data_with_gradient_3d_surface(x_grid, y_grid, f_values, figsize=(40,20)):
    """
    Visualizes data using a 3D surface plot with specified transparency and optionally plots the gradient of a function
    at a specific position, projecting the gradient vector onto the plot's base plane.

    Parameters
    ----------
    x_grid : np.array
        The grid of x values generated by np.meshgrid.
    y_grid : np.array
        The grid of y values generated by np.meshgrid.
    f_values : np.array
        The function values at each point in the grid, shape (n_samples, n_samples).
    title : str
        The title for the plot.

    Returns
    -------
    matplotlib.pyplot
        The plot object.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with specified alpha for transparency
    surf = ax.plot_surface(x_grid, y_grid, f_values, cmap='coolwarm', edgecolor='none', alpha=0.5)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Function value')

    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    #ax.set_zlabel('Function value')

    return fig


def plot_data_with_gradient_2d_contour(x_grid, y_grid, f_values, avg_x_position=None, grad_f_at_avg_x=None, figsize=(15, 10)):
    """
    Visualizes data using a 2D contour plot and optionally plots the gradient of a function at a specific position.

    Parameters
    ----------
    x_grid : np.array
        The grid of x values generated by np.meshgrid.
    y_grid : np.array
        The grid of y values generated by np.meshgrid.
    f_values : np.array
        The function values at each point in the grid, shape (n_samples, n_samples).
    avg_x_position : np.array, optional
        The average x position for plotting the gradient vector.
    grad_f_at_avg_x : np.array, optional
        The gradient of the function at the average x position.
    title : str
        The title for the plot.

    Returns
    -------
    matplotlib.pyplot
        The plot object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot isolines using the function values
    contour = ax.contour(x_grid, y_grid, f_values, levels=10, cmap='coolwarm')
    ax.clabel(contour, inline=True, fontsize=8)
    fig.colorbar(contour, ax=ax, label='Function value')
    
    # Plot gradient as vector at the average x position, if provided
    if avg_x_position is not None and grad_f_at_avg_x is not None:
        ax.quiver(*avg_x_position, *grad_f_at_avg_x, color='black', scale=1, scale_units='xy', angles='xy', width=0.005, label='Gradient Vector')
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.legend()

    return fig


def visualize_data(X1, X2, f_grid, xbar=None, grad_true=None):
    """
    Wrapper for plotting the true function

    Parameters
    ----------
    X1 : np.ndarray
        x1-values from grid generating function
    X2 : np.ndarray
        x2-values from grid generating function
    f_grid : np.ndarray
        y-values from grid generating function
    xbar : np.ndarray, optional
        local treatment point of interest, by default None
    grad_true : np.ndarray, optional
        true gradient at treatment point of interest, by default None

    Returns
    -------
    tuple of figures
        (2d-plot, 3d-surface plot)
    """
    fig_2d = plot_data_with_gradient_2d_contour(X1, X2, f_grid, avg_x_position=xbar, grad_f_at_avg_x=grad_true)
    fig_surface = plot_data_with_gradient_3d_surface(X1, X2, f_grid, avg_x_position=xbar, grad_f_at_avg_x=grad_true)
    return fig_2d, fig_surface


def visualize_gradients(upper_bounds, lower_bounds, grad_true, len_ = None, range_ = 50):
    """
    Visualize gradient for one seed

    Parameters
    ----------
    upper_bounds : np.ndarray
        upper bound for gradient for one seed for different strategies
    lower_bounds : np.ndarray
        lower bound for gradient for one seed for different strategies
    grad_true : np.ndarray
        true gradient
    len_ : int, optional
        length of experiments, by default None
    range_ : int, optional
        range around true gradient value, range of y-axis, by default 50

    Returns
    -------
    _type_
        _description_
    """
    fig = go.Figure()
    
    for i, key in enumerate(lower_bounds.keys()):

        if len_ == None:
            len_ = len(lower_bounds[key])
        # Change scatter to lines and use different line shapes for lower and upper bounds
        try: 
            fig.add_trace(go.Scatter(x=list(range(len_)), y=lower_bounds[key][:len_], mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[0]), name=f"Lower-{key}", showlegend=True))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=upper_bounds[key][:len_], mode='lines', line=dict(color=colors[i % len(colors)], dash=shapes[1]), name=f"Upper-{key}", showlegend=True))
            fig.add_trace(go.Scatter(x=list(range(len_)), y=grad_true[:len_], mode='lines', line=dict(color='black'), name='True Gradient', showlegend=False))
        except IndexError:
            print(f"IndexError: {key}")
    # Calculate the mean of grad_true
    if range_ is not None:
        mean_grad_true = np.mean(grad_true)
        fig.update_yaxes(range=[mean_grad_true - range_, mean_grad_true + range_])
        
    # Remove the legend for the upper bounds
    fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=1.0
    )
    )
    
    return fig


def plot_data_2d(x, f, xbar=None, grad_true=None):
    """
    Plots data for 2d functions.

    Parameters
    ----------
    x : np.ndarray
        samples of x values
    f : function
        function to evaluate at each point in the grid
    xbar : np.ndarray, optional
        local treatment point, by default None
    grad_true : np.ndarray, optional
        gradient, resp. value of functional at treatment point, by default None

    Returns
    -------
    matplotlib.pyplot, matplotlib.pyplot
        Visualiation of the function in 2d and 3d
    """
    X1, X2, _, f_grid = generate_grid(x, f)
    fig_2d = plot_data_with_gradient_2d_contour(X1, X2, f_grid, avg_x_position=xbar, grad_f_at_avg_x=grad_true)
    fig_surface = plot_data_with_gradient_3d_surface(X1, X2, f_grid, avg_x_position=xbar, grad_f_at_avg_x=grad_true)
    return fig_2d, fig_surface


def generate_grid(x, f, n_samples=200):
    """
    Generates grid of x values and corresponding function values for visualization purposes.

    Parameters
    ----------
    x : np.ndarray
        samples of x values
    f : function
        function to evaluate at each point in the grid
    n_samples : int, optional
        number of samples, by default 200

    Returns
    -------
    tuple
        (X1, X2, x_grid, f_grid)
    """

    # Generate a grid of points
    x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), n_samples)
    x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), n_samples)
    X1, X2 = np.meshgrid(x1, x2)
    x_grid = np.column_stack([X1.ravel(), X2.ravel()])
    f_grid = f(x_grid).reshape(X1.shape)
    return X1, X2, x_grid, f_grid