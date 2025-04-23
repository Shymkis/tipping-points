import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, root_scalar
import seaborn as sns
from sklearn.metrics import r2_score
import simulation
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- SAVE AND LOAD DATA ---

DATA_FILE = "simulation_data.csv"

def load_data(filename=DATA_FILE):
    """Load data from a CSV file into a pandas DataFrame, converting string representations of lists to numpy arrays."""
    def mutate(row):
        row["f_values"] = np.array(ast.literal_eval(row["f_values"]))
        row["A_values"] = np.array(ast.literal_eval(row["A_values"]))
        row["ord_A_values"] = np.array(ast.literal_eval(row["ord_A_values"]))
        row["popt"], row["perr"] = fit_best_model(row["f_values"], row["A_values"])
        row["A_fit"] = objective(row["f_values"], *row["popt"])
        row["fstar"] = get_fstar(row["popt"])
        row["curve_ci"], row["fstar_ci"] = get_ci(row["f_values"], *row["popt"], *row["perr"])
        return row
    df = pd.read_csv(filename)
    df = df.apply(lambda row: mutate(row), axis=1)
    return df

def save_data(df: pd.DataFrame, filename=DATA_FILE):
    """Save data from a pandas DataFrame to a CSV file, converting numpy arrays to lists."""
    def mutate(row):
        if isinstance(row["f_values"], np.ndarray): row["f_values"] = row["f_values"].tolist()
        if isinstance(row["A_values"], np.ndarray): row["A_values"] = row["A_values"].tolist()
        if isinstance(row["ord_A_values"], np.ndarray): row["ord_A_values"] = row["ord_A_values"].tolist()
        return row
    df = df.drop(columns=["popt", "perr", "A_fit", "fstar", "curve_ci", "fstar_ci"])
    df = df.apply(lambda row: mutate(row), axis=1)
    df.sort_values(by=["N", "topology", "torus", "moore", "k", "memory", "influencer_placement", "mobility_rate", "steps"], inplace=True)
    df.to_csv(filename, index=False)

def collect_data(df: pd.DataFrame, filename=DATA_FILE):
    """Collect data from simulations and save to a CSV file, skipping existing configurations."""
    from itertools import product
    for p in product([0, .01, .05, .1, .25, .5], [2, 4, 6, 8]):
        config = {
            "N": 100,
            "topology": "small-world",
            "torus": "False",
            "moore": "False",
            "k": p[1],
            "memory": 5,
            "influencer_placement": "even",
            "mobility_rate": p[0],
            "steps": 200
        }
        if not df[
              (df["N"] == config["N"]) & 
              (df["topology"] == config["topology"]) & 
              (df["torus"] == config["torus"]) & 
              (df["moore"] == config["moore"]) & 
              (df["k"] == config["k"]) & 
              (df["memory"] == config["memory"]) & 
              (df["influencer_placement"] == config["influencer_placement"]) & 
              (df["mobility_rate"] == config["mobility_rate"]) & 
              (df["steps"] == config["steps"])
              ].empty:
            print("Skipping existing configuration:", config)
            continue
        f_values = np.linspace(0, .5, 50)
        data = np.array([simulation.run(f, **config) for f in f_values])
        A_values = data[:, 0]
        ord_A_values = data[:, 1]
        df.loc[len(df)] = [
            *config.values(),
            f_values.tolist(),
            A_values.tolist(),
            ord_A_values.tolist(),
            0, 0, 0, 0, 0, 0 # Placeholders for popt, perr, A_fit, fstar, curve_ci, fstar_ci
        ]
    save_data(df, filename)

# --- MODEL FUNCTIONS ---

def logistic(f, a, b):
    """Models the adoption rate of ordinary agents."""
    return (1/(1 + np.exp(-a*(f - b))))

def objective(f, a, b):
    """Models the adoption rate of all agents."""
    return f + (1 - f)*logistic(f, a, b)

def fit_best_model(f_values, A_values, func=objective):
    """Fit a function to simulation data and return parameters + confidence intervals."""
    popt, pcov = curve_fit(func, f_values, A_values, p0=[100, 0.1], maxfev=5000, bounds=[[0, 0], [500, 1]])
    perr = np.sqrt(np.diag(pcov))  # Standard errors of parameters
    return popt, perr

def get_fstar(popt, func=objective):
    """Find tipping point given optimal function parameters."""
    popt[0] = max(popt[0], 0)  # Ensure a is non-negative
    return root_scalar(lambda f: func(f, *popt) - 0.5, bracket=[0, 1]).root  # Tipping point is where A(f) = 0.5

def get_ci(f_values, a_fit, b_fit, a_se, b_se, func=objective):
    """Calculate the confidence interval for the fitted curve using parameter errors."""
    curves = []
    fstars = []
    for a_var in [a_fit - 1.96*a_se, a_fit + 1.96*a_se]:
        for b_var in [b_fit - 1.96*b_se, b_fit + 1.96*b_se]:
            curves.append(func(f_values, a_var, b_var))
            fstars.append(get_fstar([a_var, b_var], func))
    curves = np.array(curves)
    curve_ci = (np.min(curves, axis=0), np.max(curves, axis=0))
    fstar_ci = (np.min(fstars), np.max(fstars))
    return curve_ci, fstar_ci

# --- PLOTTING FUNCTIONS ---

def plot_A_f_curve(row):
    """Plot A(f) vs. f with fitted curve and confidence intervals."""
    config = row[:9].to_dict()
    f_values = row["f_values"]
    A_values = row["A_values"]
    A_fit = row["A_fit"]
    popt = row["popt"]
    fstar = row["fstar"]
    curve_lower, curve_upper = row["curve_ci"]
    fstar_lower, fstar_upper = row["fstar_ci"]
    plt.figure(figsize=(8, 5))
    plt.scatter(f_values, A_values, label="Simulation Data", alpha=.6) # Scatter points
    plt.plot(f_values, A_fit, label=f"Fitted Curve A(f)", color="red") # Fitted curve
    plt.fill_between(f_values, curve_lower, curve_upper, color="red", alpha=.2, label="A(f) 95% CI") # Curve confidence interval
    plt.axhline(0.5, linestyle="--", color="gray", alpha=.7) # Tipping point hline
    plt.axvline(fstar, linestyle="--", color="green", label=f"Tipping Point f*: {fstar:.3f}") # Tipping point vline
    plt.axvspan(fstar_lower, fstar_upper, color="green", alpha=.2, label=f"f* 95% CI: [{fstar_lower:.3f}, {fstar_upper:.3f}]") # Tipping point confidence interval
    plt.xlabel("Proportion of Influencers (f)")
    plt.ylabel("Final Adoption Rate (A(f))")
    plt.title(list(config.values()))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_curves(df: pd.DataFrame):
    df.apply(lambda row: plot_A_f_curve(row), axis=1)

def plot_best_curves_first(df: pd.DataFrame):
    """Plot the best curves for each configuration."""
    df["SSR"] = df.apply(lambda row: np.sum(
        (np.array(row["A_values"]) - objective(
            np.array(row["f_values"]),
            *row["popt"]
        ))**2),
        axis=1
    )
    df["logistic_SSR"] = df.apply(lambda row: np.sum(
        (np.array(row["A_values"]) - logistic(
            np.array(row["f_values"]),
            *fit_best_model(row["f_values"], row["A_values"], logistic)[0]
        ))**2),
        axis=1
    )
    df["diff"] = df["logistic_SSR"] - df["SSR"]
    df["better"] = df["diff"] > 0
    print(df["diff"].describe())
    print(df["better"].value_counts())
    df.sort_values(by="diff", inplace=True, ascending=False)
    plot_all_curves(df.head())

def plot_fstar_heatmaps(df, vars=["mobility_rate", "N", "steps", "memory", "topology"]):
    """Generate heatmaps to visualize the relationship between configuration parameters and the tipping point (f*)."""
    fig, axes = plt.subplots(len(vars), len(vars), figsize=(10, 10))
    for i in range(len(vars)):
        y = vars[i]
        for j in range(len(vars)):
            x = vars[j]
            ax: plt.Axes = axes[i, j]
            # Put axis labels on the diagonal
            if i == j:
                ax.text(0.5, 0.5, y, ha='center', va='center', fontsize=12)
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            # Plot heatmap
            heatmap_df = df.pivot_table(index=y, columns=x, values="fstar", aggfunc="mean")
            sns.heatmap(heatmap_df, ax=ax, cmap="Greens", cbar=False, annot=True, fmt=".2f")
            # Remove redundant axis labels
            ax.set_xlabel("")
            ax.set_ylabel("")
            # X-axis ticks
            if i == 0:
                ax.set_xticklabels(heatmap_df.columns, rotation=30)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
            elif i == len(vars) - 1:
                ax.set_xticklabels(heatmap_df.columns, rotation=30)
            else:
                ax.set_xticks([])
            # Y-axis ticks
            if j == 0:
                ax.set_yticklabels(heatmap_df.index, rotation=30)
            elif j == len(vars) - 1:
                ax.set_yticklabels(heatmap_df.index, rotation=30)
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            else:
                ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_fstar_trendlines(df: pd.DataFrame, x, hue=None, size=None, style=None, func=None):
    """Plot trendlines of f* against mobility rate for different configurations."""
    plt.figure(figsize=(5,5))
    sns.lineplot(data=df, x=x, y="fstar", hue=hue, size=size, style=style)
    plt.xlabel(x)
    plt.ylabel("Tipping Point (f*)")
    plt.title(f"Tipping Point Trends: f* vs. {x}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fstar_memory(df: pd.DataFrame):
    """Plot the relationship between memory size and the tipping point (f*)."""
    memory_values = df["memory"]
    f_star_values = df["fstar"]
    df_pos = df[df["memory"] > 0]
    memory_values_pos = df_pos["memory"]
    f_star_values_pos = df_pos["fstar"]

    def log_func(x, a, b):
        return a * np.log(x) + b
    params_log, _ = curve_fit(log_func, memory_values_pos, f_star_values_pos)
    r2_log = r2_score(f_star_values_pos, log_func(memory_values_pos, *params_log))
    print(f"R² value: {r2_log:.4f}")

    def linear_func(x, m, c):
        return m * x + c
    params_linear, _ = curve_fit(linear_func, memory_values, f_star_values)
    r2_linear = r2_score(f_star_values, linear_func(memory_values, *params_linear))
    print(f"Linear R²: {r2_linear:.4f}")

    def exp_func(x, A, B):
        return A * np.exp(-B * x)
    params_exp, _ = curve_fit(exp_func, memory_values, f_star_values)
    r2_exp = r2_score(f_star_values, exp_func(memory_values, *params_exp))
    print(f"Exponential R²: {r2_exp:.4f}")

    def log_shifted_func(x, a, b):
        return a * np.log(x + b)
    params_log_shifted, cov = curve_fit(log_shifted_func, memory_values, f_star_values)
    r2_log_shifted = r2_score(f_star_values, log_shifted_func(memory_values, *params_log_shifted))
    print(f"Log-Shifted R²: {r2_log_shifted:.4f}")

    def exp_shifted_func(x, A, B, C):
        return -A * np.exp(-B * (x + C)) + .5
    params_exp_shifted, _ = curve_fit(exp_shifted_func, memory_values, f_star_values)
    r2_exp_shifted = r2_score(f_star_values, exp_shifted_func(memory_values, *params_exp_shifted))
    print(f"Exponential-Shifted R²: {r2_exp_shifted:.4f}")

    def reciprocal_func(x, A, B, C):
        return A / (x**B + C)
    params_reciprocal, cov_reciprocal = curve_fit(reciprocal_func, memory_values_pos, f_star_values_pos)
    r2_reciprocal = r2_score(f_star_values_pos, reciprocal_func(memory_values_pos, *params_reciprocal))
    print(f"Reciprocal R²: {r2_reciprocal:.4f}")

    # Plotting the fitted curves
    memory_fit = np.linspace(memory_values.min(), memory_values.max(), 100)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df, x="memory", y="fstar", hue="topology")
    sns.lineplot(x=memory_fit, y=log_shifted_func(memory_fit, *params_log_shifted), label=f"Shifted Log Fit (R²={r2_log_shifted:.4f})", color="purple")
    log_shifted_ci = np.array([
        log_shifted_func(memory_fit, *(params_log_shifted + np.sqrt(np.diag(cov)))),
        log_shifted_func(memory_fit, *(params_log_shifted - np.sqrt(np.diag(cov))))
    ])
    plt.fill_between(memory_fit, log_shifted_ci[0], log_shifted_ci[1], color="purple", alpha=0.2, label="95% CI")
    plt.xlabel("Memory Size")
    plt.ylabel("Tipping Point (f*)")
    plt.title("Tipping Point vs. Memory Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fstar_steps(df: pd.DataFrame):
    """Plot the relationship between maximum number of steps and the tipping point (f*)."""
    step_values = df["steps"]
    f_star_values = df["fstar"]

    def log_func(x, a, b):
        return a * np.log(x) + b
    params_log, cov_log = curve_fit(log_func, step_values, f_star_values)
    r2_log = r2_score(f_star_values, log_func(step_values, *params_log))
    print(f"Log R² value: {r2_log:.4f}")

    def linear_func(x, m, c):
        return m * x + c
    params_linear, cov_linear = curve_fit(linear_func, step_values, f_star_values)
    r2_linear = r2_score(f_star_values, linear_func(step_values, *params_linear))
    print(f"Linear R²: {r2_linear:.4f}")

    def exp_func(x, A, B):
        return A * np.exp(-B * x)
    params_exp, cov_exp = curve_fit(exp_func, step_values, f_star_values, p0=[.25, .01])
    r2_exp = r2_score(f_star_values, exp_func(step_values, *params_exp))
    print(f"Exponential R²: {r2_exp:.4f}")

    def reciprocal_func(x, A, B):
        return A / (x**B)
    params_reciprocal, cov_reciprocal = curve_fit(reciprocal_func, step_values, f_star_values)
    r2_reciprocal = r2_score(f_star_values, reciprocal_func(step_values, *params_reciprocal))
    print(f"Reciprocal R²: {r2_reciprocal:.4f}")

    # Plotting the fitted curves
    step_fit = np.linspace(step_values.min(), step_values.max(), 100)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=df, x="steps", y="fstar", hue="topology")
    sns.lineplot(x=step_fit, y=reciprocal_func(step_fit, *params_reciprocal), label=f"Reciprocal Fit (R²={r2_reciprocal:.4f})", color="maroon")
    exp_ci = np.array([
        reciprocal_func(step_fit, *(params_reciprocal + np.sqrt(np.diag(cov_reciprocal)))),
        reciprocal_func(step_fit, *(params_reciprocal - np.sqrt(np.diag(cov_reciprocal))))
    ])
    plt.fill_between(step_fit, exp_ci[0], exp_ci[1], color="maroon", alpha=0.2, label="95% CI")
    plt.xlabel("Number of Steps")
    plt.ylabel("Tipping Point (f*)")
    plt.title("Tipping Point vs. Maximum Number of Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- HIGH-LEVEL ANALYSIS FUNCTIONS ---

def run_anova(df: pd.DataFrame):
    """Run ANOVA on the dataset to analyze the impact of different factors on f*."""
    model = ols("fstar ~ C(topology) + N + memory + mobility_rate + steps", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

def param_correlations(df: pd.DataFrame):
    """Calculate and print the correlation matrix for the parameters in the dataset."""
    df_encoded = df.copy()
    df_encoded["topology"] = df_encoded["topology"].map({"lattice": 0, "small-world": 1})
    corr_matrix = df_encoded[["N", "topology", "memory", "mobility_rate", "steps", "fstar"]].corr("spearman")
    print(corr_matrix)

if __name__ == "__main__":
    df = load_data()

    # collect_data(df)

    subset = df[
        (df["N"] == 100) &
        (df["topology"] == "small-world") &
        (df["torus"] == False) &
        (df["moore"] == False) &
        (df["k"].isin([2, 4, 6, 8])) &
        (df["memory"] == 5) &
        (df["steps"] == 200) &
        (df["mobility_rate"].isin([0, .1, .25]))
    ]
    subset.sort_values(by=["mobility_rate", "k"], inplace=True)
    print(subset.loc[:, ["mobility_rate", "k", "fstar", "fstar_ci"]])

    # plot_best_curves_first(subset)
    # plot_fstar_heatmaps(subset)
    # param_correlations(subset)
    # plot_fstar_trendlines(subset, x="mobility_rate", hue="steps", style="memory")
    # plot_fstar_memory(subset)
    # plot_fstar_steps(subset)
    # run_anova(subset)
