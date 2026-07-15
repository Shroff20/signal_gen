import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import sympy as sp




def generate_samples(config, N = 1, mode = 'LHS', seed = 0):

    # Generate samples for each signal based on the configuration
    np.random.seed(seed)
    generation_keys = ['signals', 'signal_generation']

    samples = {}

    for key in generation_keys:
        samples[key] = {}

        for signal_name, data in config[key].items():
            samples[key][signal_name] = {}

            for param_name, d in data['parameters'].items():
                distribution_type = d['distribution_type']
                if distribution_type == 'uniform':

                    if mode == 'random':
                        sample_vec = np.random.uniform(d["min_value"], d["max_value"], size = N)
                    elif mode == 'LHS':
                        rng = np.random.default_rng(seed)
                        sample_vec = np.linspace(d["min_value"], d["max_value"], N)
                        rng.shuffle(sample_vec)
                    else:
                        raise ValueError(f"mode={mode} is not recognized")
                    
                    samples[key][signal_name][param_name] = sample_vec
                
                else:
                    raise ValueError(f"distribution_type={distribution_type} is not recognized")

    return samples






def _clip_signal(t, y, min_val, max_val):
    # Add timepoints where signal crosses min/max limits
    
    # Find segments that cross boundaries
    crossing_points_t = []
    crossing_points_y = []
    
    for i in range(len(t) - 1):
        y_start, y_end = y[i], y[i + 1]
        t_start, t_end = t[i], t[i + 1]
        dt = t_end - t_start
        
        if dt == 0:
            continue
        
        slope = (y_end - y_start) / dt
        
        # Check max_val crossing
        if (y_start < max_val < y_end) or (y_end < max_val < y_start):
            t_cross = t_start + (max_val - y_start) / slope
            crossing_points_t.append(t_cross)
            crossing_points_y.append(max_val)
        
        # Check min_val crossing
        if (y_start < min_val < y_end) or (y_end < min_val < y_start):
            t_cross = t_start + (min_val - y_start) / slope
            crossing_points_t.append(t_cross)
            crossing_points_y.append(min_val)
    
    # Merge crossing points with original points
    if crossing_points_t:
        t = np.concatenate([t, np.array(crossing_points_t)])
        y = np.concatenate([y, np.array(crossing_points_y)])
        
        # Sort by time
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        y = y[sort_idx]
    
    y = np.clip(y, min_val, max_val)

    return t, y


def _generate_LHS_samples(config, N):





    pass 


def _generate_ramp_signal(signal_name, signal_config, ramp_config, end_time_pad=1):

    initial_value = _get_sample(signal_config)
    ramp_start_time = _get_sample(ramp_config["ramp_start_time"])
    ramp_duration = _get_sample(ramp_config["ramp_duration"])
    end_time = _get_sample(ramp_config["end_time"])
    hold_duration = _get_sample(ramp_config["hold_duration"])

    if np.random.uniform(0, 1) > signal_config.get("no_change_probability", 0):
        ramp_rate = np.random.uniform(
            -signal_config["max_rate_of_change"], signal_config["max_rate_of_change"]
        )
    else:
        ramp_rate = 0

    final_value = initial_value + ramp_rate * ramp_duration

    t0 = 0
    t1 = ramp_start_time
    t2 = t1 + ramp_duration
    t3 = t2 + hold_duration
    t4 = t3 + ramp_duration
    t5 = np.max([t4 + end_time_pad, end_time])

    y0 = initial_value
    y1 = initial_value
    y2 = final_value
    y3 = final_value
    y4 = initial_value
    y5 = initial_value

    t = np.array([t0, t1, t2, t3, t4, t5])
    y = np.array([y0, y1, y2, y3, y4, y5])

    t, y = _clip_signal(t, y, signal_config["min_value"], signal_config["max_value"])

    df = pd.DataFrame(index=t, data={signal_name: y})

    if end_time not in df.index:
        df.loc[end_time] = np.nan
        df = df.sort_index()
        df = df.interpolate(method="index")

    I = df.index <= ramp_config["end_time"]['value']
    df = df.loc[I]

    signal_range = df[signal_name].max() - df[signal_name].min()
    effective_ramp_duration = np.abs(signal_range / ramp_rate) if ramp_rate != 0 else 0

    metadata = {
        "initial_value": initial_value,
        "ramp_start_time": ramp_start_time,
        "effective_ramp_duration": effective_ramp_duration,
        "ramp_rate": ramp_rate,
        "hold_duration": hold_duration,
        "signal_range": signal_range,
    }

    return df, metadata


def _add_derived_signals(df_merged, config):

    for signal in config['derived_signals'].keys():

        eqn = config['derived_signals'][signal]['equation']
        f = sp.sympify(eqn)

        variables = sorted(list(f.free_symbols), key=lambda s: s.name)
        numerical_func = sp.lambdify(variables, f, 'numpy')
        variables_str = [str(v) for v in variables]

        inputs = [df_merged[variable].to_numpy() for variable in variables_str]
        derived_signal_values = numerical_func(*inputs)

        df_merged[signal] = derived_signal_values

    return df_merged


def generate_ramp_loadcase(config):

    dfs = []
    metadatas = {}

    for signal_name, signal_config in config["signals"].items():
        df, metadata = _generate_ramp_signal(
            signal_name, signal_config, config["ramp_config"]
        )
        dfs.append(df)
        metadatas[signal_name] = metadata

    df_merged = pd.concat(dfs, axis=1)
    df_merged = df_merged.sort_index()
    df_merged = df_merged.interpolate(method="index")

    I = df_merged.index <= config["ramp_config"]["end_time"]['value']
    df_merged = df_merged.loc[I]

    df_parameters = pd.DataFrame.from_dict(metadatas, orient="index")


    df_merged = _add_derived_signals(df_merged, config)
    df_merged.index.name = "t"


    return df_merged, df_parameters


def load_config(fn_yaml):
    with open(fn_yaml, "r") as f:
        config = yaml.safe_load(f)

    pprint(config, sort_dicts=False)
    return config


def plot_loadcase(df_merged, config, loadcase_name = None):

    fig, ax = plt.subplots(len(df_merged.columns))
    for i, signal_name in enumerate(df_merged.columns):


        if signal_name in config['signals']:
            plot_color = config['signals'][signal_name].get('plot_color', 'gray')
            yrange = config['signals'][signal_name]['max_value'] - config['signals'][signal_name]['min_value']
            plot_min = config['signals'][signal_name]['min_value'] - 0.05 * yrange
            plot_max = config['signals'][signal_name]['max_value'] + 0.05 * yrange
            ax[i].plot(df_merged.index, df_merged[signal_name], label = f'{signal_name}', linewidth = 2, c= plot_color)
            ax[i].set_ylim(plot_min, plot_max)

        elif signal_name in config['derived_signals']:
            plot_color = config['derived_signals'][signal_name].get('plot_color', 'gray')
            ax[i].plot(df_merged.index, df_merged[signal_name], label = f'{signal_name}\n(derived)', linewidth = 2, c= plot_color)

        ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle(f"{loadcase_name}")
    fig.tight_layout()