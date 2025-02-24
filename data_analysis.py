import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, root_scalar
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- SAVE AND LOAD DATA ---

DATA_FILE = "simulation_data.csv"

def load_data(filename=DATA_FILE):
    """Load data from a CSV file into a pandas DataFrame, converting specific columns from strings to lists."""
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

# --- MODEL FUNCTIONS ---

def logistic(f, a, b):
    """Models the adoption rate of ordinary agents."""
    return (1/(1 + np.exp(-a*(f - b))))

def objective(f, a, b):
    """Models the adoption rate of all agents."""
    return f + (1 - f)*logistic(f, a, b)

def fit_best_model(f_values, A_values, func=objective):
    """Fit a function to simulation data and return parameters + confidence intervals."""
    popt, pcov = curve_fit(func, f_values, A_values, p0=[100, 0.1], maxfev=5000, bounds=[[0, 0], [1000, 1]])
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

def plot_fstar_heatmaps(df, vars=["mobility_rate", "N", "steps", "memory", "topology"]):
    """Generate heatmaps to visualize the relationship between configuration parameters and the tipping point (f*)."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
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
            elif i == 4:
                ax.set_xticklabels(heatmap_df.columns, rotation=30)
            else:
                ax.set_xticks([])
            # Y-axis ticks
            if j == 0:
                ax.set_yticklabels(heatmap_df.index, rotation=30)
            elif j == 4:
                ax.set_yticklabels(heatmap_df.index, rotation=30)
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            else:
                ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_fstar_trendlines(df: pd.DataFrame, x, hue=None, size=None, style=None):
    """Plot trendlines of f* against mobility rate for different configurations."""
    plt.figure(figsize=(5,5))
    sns.lineplot(data=df, x=x, y="fstar", hue=hue, size=size, style=style)
    plt.xlabel(x)
    plt.ylabel("Tipping Point (f*)")
    plt.title(f"Tipping Point Trends: f* vs. {x}")
    plt.legend()
    plt.show()

def plot_all_curves(df: pd.DataFrame):
    df.apply(lambda row: plot_A_f_curve(row), axis=1)

def plot_best_curves_first(df: pd.DataFrame):
    """Plot the best curves for each configuration."""
    df["residuals"] = df.apply(lambda row: np.sum(
        (np.array(row["A_values"]) - objective(
            np.array(row["f_values"]),
            *row["popt"]
        ))**2),
        axis=1
    )
    df["logistic_residuals"] = df.apply(lambda row: np.sum(
        (np.array(row["A_values"]) - logistic(
            np.array(row["f_values"]),
            *fit_best_model(row["f_values"], row["A_values"], logistic)[0]
        ))**2),
        axis=1
    )
    df["diff"] = df["logistic_residuals"] - df["residuals"]
    df["better"] = df["diff"] > 0
    print(df["diff"].describe())
    print(df["better"].value_counts())
    df.sort_values(by="diff", inplace=True, ascending=False)
    plot_all_curves(df.head())

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
    corr_matrix = df_encoded[["N", "topology", "memory", "mobility_rate", "steps", "fstar"]].corr()
    print(corr_matrix)

if __name__ == "__main__":
    df = load_data()
    df = df[(df["torus"] == False) & (df["moore"] == False) & (df["k"] == 4)]

    plot_best_curves_first(df)
    param_correlations(df)
    run_anova(df)
    plot_fstar_heatmaps(df)
    plot_fstar_trendlines(df, x="mobility_rate", hue="steps", style="memory")
