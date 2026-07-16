import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import sympy as sp

np.random.seed(0)


class SignalGenerator:

    def __init__(self):

        self.config = None
        self.df_parameter_values = None
        self.df_loadcases = None

    def load_config(self, config_filename):
        self.config = _load_config(config_filename)
        print(f' * loaded {config_filename}')

    def sample_parameter_distributions(self, N_samples, mode="LHS", seed=None):
        self.df_parameter_values = _generate_samples(
            self.config, N=N_samples, mode=mode, seed=seed
        )
        print(f' * generated {N_samples=:} sets of parameter realizations using {mode=:}')


    def generate_signals(self):

        N_loadcases = len(self.df_parameter_values)

        df_loadcases = pd.DataFrame(
            index=range(N_loadcases),
            columns=["loadcase_name", "df_signals", "df_metadata"],
        )

        for i in range(N_loadcases):
            loadcase_name = f"ramp_{i+1:05d}"
            df_signals, df_metadata = _generate_ramp_signals(
                self.df_parameter_values, self.config, i
            )

            df_loadcases.loc[i, ["loadcase_name", "df_signals", "df_metadata"]] = [
                loadcase_name,
                df_signals,
                df_metadata,
            ]

        print(f' * generated {N_loadcases=:} loadcases, each with {len(df_signals.columns)} signals')


        self.df_loadcases = df_loadcases

    def plot_loadcase(self, idx):

        df_signals = self.df_loadcases.loc[idx, "df_signals"]
        loadcase_name = self.df_loadcases.loc[idx, "loadcase_name"]

        print(f' * plotting {loadcase_name} (idx = {idx}):')

        _plot_loadcase(df_signals, self.config, loadcase_name=loadcase_name)


def _generate_samples(config, N=1, mode="LHS", seed=None):

    # Generate samples for each signal based on the configuration

    signals = config["signals"].keys()
    samples = {}

    for signal in signals:
        samples[signal] = {}

        for param_name, d in config["signals"][signal]["parameters"].items():
            distribution_type = d["distribution_type"]
            if distribution_type == "uniform":

                if mode == "random":
                    sample_vec = np.random.uniform(
                        d["min_value"], d["max_value"], size=N
                    )
                elif mode == "LHS":
                    rng = np.random.default_rng(seed)
                    sample_vec = np.linspace(d["min_value"], d["max_value"], N)
                    rng.shuffle(sample_vec)
                else:
                    raise ValueError(f"mode={mode} is not recognized")

                samples[signal][param_name] = sample_vec
            else:
                raise ValueError(
                    f"distribution_type={distribution_type} is not recognized"
                )

        for param_name, d in config["signal_generation"]["ramp"]["parameters"].items():
            distribution_type = d["distribution_type"]
            if distribution_type == "uniform":
                if mode == "random":
                    sample_vec = np.random.uniform(
                        d["min_value"], d["max_value"], size=N
                    )
                elif mode == "LHS":
                    rng = np.random.default_rng(seed)
                    sample_vec = np.linspace(d["min_value"], d["max_value"], N)
                    rng.shuffle(sample_vec)
                else:
                    raise ValueError(f"mode={mode} is not recognized")

                samples[signal][param_name] = sample_vec
            else:
                raise ValueError(
                    f"distribution_type={distribution_type} is not recognized"
                )

    df_samples = pd.concat({k: pd.DataFrame(v) for k, v in samples.items()}, axis=1)
    df_samples.index.name = "loadcase"

    return df_samples


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


def _generate_ramp_signals(df_samples, config, idx):

    dfs = []
    metadatas = {}

    signal_names = df_samples.columns.levels[0]

    for signal_name in signal_names:

        initial_value = df_samples.loc[idx, (signal_name, "magnitude")]
        ramp_rate = df_samples.loc[
            idx, (signal_name, "rate_of_change")
        ]  # TODO:  add probability to set to 0

        ramp_start_time = df_samples.loc[idx, (signal_name, "ramp_start_time")]
        ramp_duration = df_samples.loc[idx, (signal_name, "ramp_duration")]
        end_time = df_samples.loc[idx, (signal_name, "end_time")]
        hold_duration = df_samples.loc[idx, (signal_name, "hold_duration")]

        min_magnitude = config["signals"][signal_name]["parameters"]["magnitude"][
            "min_value"
        ]
        max_magnitude = config["signals"][signal_name]["parameters"]["magnitude"][
            "max_value"
        ]

        df, metadata = _generate_ramp_signal(
            signal_name,
            initial_value,
            ramp_rate,
            ramp_duration,
            ramp_start_time,
            hold_duration,
            end_time,
            min_magnitude,
            max_magnitude,
        )

        dfs.append(df)
        metadatas[signal_name] = metadata

    df_signals = pd.concat(dfs, axis=1)
    df_signals = df_signals.sort_index()
    df_signals = df_signals.interpolate(method="index")

    end_time = config["signal_generation"]["ramp"]["parameters"]["end_time"][
        "min_value"
    ]
    df_signals = df_signals.loc[df_signals.index <= end_time]

    df_parameters = pd.DataFrame.from_dict(metadatas, orient="index")

    df_signals = _add_derived_signals(df_signals, config)
    df_signals.index.name = "t"

    return df_signals, df_parameters


def _generate_ramp_signal(
    signal_name,
    initial_value,
    ramp_rate,
    ramp_duration,
    ramp_start_time,
    hold_duration,
    end_time,
    min_magnitude,
    max_magnitude,
    end_time_pad=1,
):

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

    t, y = _clip_signal(t, y, min_magnitude, max_magnitude)

    df = pd.DataFrame(index=t, data={signal_name: y})
    df = df[~df.index.duplicated(keep="first")]

    if end_time not in df.index:
        df.loc[end_time] = np.nan
        df = df.sort_index()
        df = df.interpolate(method="index")

    I = df.index <= end_time
    df = df.loc[I]

    signal_range = df[signal_name].max() - df[signal_name].min()
    effective_ramp_duration = np.abs(signal_range / ramp_rate) if ramp_rate != 0 else 0

    metadata = {
        "initial_value": initial_value,
        "ramp_start_time": ramp_start_time,
        "ramp_duration": ramp_duration,
        "effective_ramp_duration": effective_ramp_duration,
        "ramp_rate": ramp_rate,
        "hold_duration": hold_duration,
        "signal_range": signal_range,
    }

    return df, metadata


def _add_derived_signals(df_merged, config):

    for signal in config["derived_signals"].keys():

        eqn = config["derived_signals"][signal]["equation"]
        f = sp.sympify(eqn)

        variables = sorted(list(f.free_symbols), key=lambda s: s.name)
        numerical_func = sp.lambdify(variables, f, "numpy")
        variables_str = [str(v) for v in variables]

        inputs = [df_merged[variable].to_numpy() for variable in variables_str]
        derived_signal_values = numerical_func(*inputs)

        df_merged[signal] = derived_signal_values

    return df_merged


def _load_config(fn_yaml):
    with open(fn_yaml, "r") as f:
        config = yaml.safe_load(f)
    return config


def _plot_loadcase(df_merged, config, loadcase_name=None):

    fig, ax = plt.subplots(len(df_merged.columns))
    for i, signal_name in enumerate(df_merged.columns):

        if signal_name in config["signals"]:
            plot_color = config["signals"][signal_name].get("plot_color", "gray")

            ymin = config["signals"][signal_name]["parameters"]["magnitude"][
                "min_value"
            ]
            ymax = config["signals"][signal_name]["parameters"]["magnitude"][
                "max_value"
            ]
            yrange = ymax - ymin
            plot_min = ymin - 0.05 * yrange
            plot_max = ymax + 0.05 * yrange
            ax[i].plot(
                df_merged.index,
                df_merged[signal_name],
                label=f"{signal_name}",
                linewidth=2,
                c=plot_color,
            )
            ax[i].set_ylim(plot_min, plot_max)

        elif signal_name in config["derived_signals"]:
            plot_color = config["derived_signals"][signal_name].get(
                "plot_color", "gray"
            )
            ax[i].plot(
                df_merged.index,
                df_merged[signal_name],
                label=f"{signal_name}\n(derived)",
                linewidth=2,
                c=plot_color,
            )

        ax[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.suptitle(f"{loadcase_name}")
    fig.tight_layout()
