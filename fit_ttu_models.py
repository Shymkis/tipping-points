import os, argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns

np.seterr(divide='ignore', invalid='ignore')

# ---------- Subfunctions ----------
def f_factor(f, p):
    """f^-p"""
    return f**-p

def M_factor(M, h):
    """ln(M + h)"""
    return np.log(M + h)

def m_factor(m, g, p, h, q):
    """(m + g)^p/(m + h)^q"""
    return ((m + g)**p)/((m + h)**q)

def sigma(x):
    """sigmoid function"""
    return 1/(1 + np.exp(-x))

def G(f, M, f_c, w_f, M_c, w_M):
    """scaling function for e_d"""
    return sigma((f_c - f)/w_f)*sigma((M - M_c)/w_M)

def e_d(f, M, e_lo, e_hi, f_c, w_f, M_c, w_M):
    """exponent for d factor"""
    return e_lo + (e_hi - e_lo)*G(f, M, f_c, w_f, M_c, w_M)

def d_factor(d, f, M, e_lo, e_hi, f_c, w_f, M_c, w_M):
    """(d/d_min)^e_d(f, M)"""
    d_min = 2 - 2/225
    return (d/d_min)**e_d(f, M, e_lo, e_hi, f_c, w_f, M_c, w_M)

# ---------- TTU Models ----------
def TTU_f(params, f):
    """1 + coeff * f^-p"""
    coeff, exp_f = params
    return 1 + (coeff
                *f_factor(f, exp_f))

def TTU_f_M(params, f, M):
    """1 + coeff * f^-p * log_b(M + h)"""
    coeff, exp_f, shift_M = params
    return 1 + (coeff
                *f_factor(f, exp_f)
                *M_factor(M, shift_M))

def TTU_f_M_m(params, f, M, m):
    """1 + coeff * f^-p * log_b(M + h) * m_factor(m, g, p, h, q)"""
    coeff, exp_f, shift_M, shift_m1, exp_m1, shift_m2, exp_m2 = params
    return 1 + (coeff
                *f_factor(f, exp_f)
                *M_factor(M, shift_M)
                *m_factor(m, shift_m1, exp_m1, shift_m2, exp_m2))

def TTU_f_M_m_d(params, f, M, m, d):
    """1 + coeff * f^-p * log_b(M + h) * m_factor(m, g, p, h, q) * d^p"""
    coeff, exp_f, shift_M, shift_m1, exp_m1, shift_m2, exp_m2, e_lo, e_hi, f_c, w_f, M_c, w_M = params
    return 1 + (coeff
                *f_factor(f, exp_f)
                *M_factor(M, shift_M)
                *m_factor(m, shift_m1, exp_m1, shift_m2, exp_m2)
                *d_factor(d, f, M, e_lo, e_hi, f_c, w_f, M_c, w_M))

# ---------- Residuals ----------
def resid_f(params, y, f):
    return TTU_f(params, f) - y

def resid_f_M(params, y, f, M):
    return TTU_f_M(params, f, M) - y

def resid_f_M_m(params, y, f, m, M):
    return TTU_f_M_m(params, f, m, M) - y

def resid_f_M_m_d(params, y, f, m, M, d):
    return TTU_f_M_m_d(params, f, m, M, d) - y

# ---------- Data loading ----------
def load_all(data_dir):
    DEG = {"lat_vn":56.0/15.0,"lat_mo":1624.0/225.0,"tor_vn":4.0,"tor_mo":8.0,"sw_k2":2.0,"sw_k4":4.0,"sw_k6":6.0,"sw_k8":8.0}
    rows = []
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".csv"):
            continue
        topo = os.path.splitext(fn)[0].lower()
        df = pd.read_csv(os.path.join(data_dir, fn))
        # normalize column names
        rename = {"m":"mobility_rate","M":"memory","fail_rate":"fail rate"}
        for k,v in rename.items():
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k:v})
        need = ["f","memory","mobility_rate","mean","variance","fail rate"]
        if any(c not in df.columns for c in need):
            continue
        df = df.rename(columns={"memory":"M","mobility_rate":"m","fail rate":"fail_rate"})
        df["topology"] = topo
        df["d"] = df["topology"].map(DEG).astype(float)
        rows.append(df[["topology","d","f","M","m","mean","variance","fail_rate"]])
    if not rows:
        raise SystemExit("No CSVs found in data_dir.")
    all_df = pd.concat(rows, ignore_index=True)
    return all_df

def preprocess(df, drop_censored=True):
    df = df.copy()
    if drop_censored:
        df = df[df["fail_rate"]!=100].copy()
    # drop missing means (after censor filter)
    df = df[~df["mean"].isna()].copy()
    # variance -> fill by group median
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
    df["variance"] = df.groupby("topology")["variance"].transform(lambda s: s.fillna(s.median()))
    if df["variance"].isna().any():
        df["variance"] = df["variance"].fillna(df["variance"].median())
    # weights: successes / variance
    trials = 50.0
    successes = np.maximum(trials * (1 - df["fail_rate"].astype(float)/100.0), 1.0)
    df["w"] = successes / (df["variance"].astype(float) + 1e-9)
    return df.reset_index(drop=True)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    p_args = ap.parse_args()

    os.makedirs(p_args.out_dir, exist_ok=True)
    raw_df = load_all(p_args.data_dir)
    LAT = ["lat_mo", "lat_vn"]
    TOR = ["tor_mo", "tor_vn"]
    GRID = LAT + TOR
    SW = ["sw_k2", "sw_k4", "sw_k6", "sw_k8"]
    df = preprocess(raw_df[raw_df["topology"].isin(GRID+SW)])
    # df = preprocess(raw_df[raw_df["topology"] == "sw_k8"])

    # Prepare data for fitting
    f = df["f"].values
    M = df["M"].values
    m = df["m"].values
    d = df["d"].values
    y = df["mean"].values

    # Fit the models
    models = [TTU_f,TTU_f_M,TTU_f_M_m,TTU_f_M_m_d]
    res_fns = [resid_f,resid_f_M,resid_f_M_m,resid_f_M_m_d]
    x0 = [[0.1,0.1],
          [0.1],
          [0.2,0.2,0.1,0.1],
          [0.1,0.1,0.1,0.1,10,6]]
    args = [(y, f),
            (y, f, M),
            (y, f, M, m),
            (y, f, M, m, d)]
    thetas = []
    n = len(y)
    ss_tot = np.sum((y - np.mean(y))**2)
    def calc_aic(n, rss, k):
        return n * np.log(rss / n) + 2 * k
    for i in range(len(models)):
        model = models[i]
        res_fn = res_fns[i]
        print(f"Fitting {model.__name__}...")
        guess = np.append(thetas[i-1], x0[i]) if i > 0 else x0[i]
        theta = least_squares(res_fn, guess, args=args[i], bounds=(-1e3,1e3)).x
        thetas.append(theta)
        with np.printoptions(suppress=True, precision=4):
            print(f"Fitted parameters: {theta}")
        y_pred = model(theta, *args[i][1:])
        rss = np.sum((y - y_pred) ** 2)
        r2 = 1 - rss / ss_tot
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        aic = calc_aic(n, rss, len(theta))
        print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}, AIC: {aic:.2f}")
        if i < len(models) - 1:
            continue
        # 3D plot of data and surface
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=45)  # Set elevation and azimuthal angle
        scatter = ax.scatter(
            f, m, y,
            s=20 + 80 * (M - np.min(M)) / (np.max(M) - np.min(M) + 1e-9),  # scale degree for size
            c=d,
            cmap='viridis',
            alpha=.2
        )
        # Create grid for surface
        # f_grid = np.linspace(np.min(f), np.max(f), 100)
        # m_grid = np.linspace(np.min(m), np.max(m), 100)
        # F, M_ = np.meshgrid(f_grid, m_grid)
        # M_mean = np.mean(M)
        # D_mean = np.mean(d)
        # Y_surf = model(thetas[-1], F, M_, np.full_like(F, M_mean), np.full_like(F, D_mean))
        # ax.plot_surface(F, M_, Y_surf, alpha=0.2, color='gray', edgecolor='none')
        ax.set_xlabel('f', labelpad=0)
        ax.set_ylabel('m', labelpad=0)
        ax.set_zlabel('T', labelpad=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1000)
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        # cbar.set_label('M (memory)')
        plt.title("all_data", y=1.05)
        plt.tight_layout()
        plt.show()
        plt.close()
        # Scatterplot of predicted vs actual
        residuals = y - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, color='skyblue')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlim(-200, 200)
        plt.title(f"T(f,M,m,d) Residuals Histogram/Density")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        plt.close()
        # Scatterplots of residuals vs f, m, M, d
        features = [("f", f), ("m", m), ("M", M), ("d", d)]
        for name, vals in features:
            plt.figure(figsize=(6, 4))
            plt.scatter(vals, residuals, alpha=0.1)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel(name)
            plt.ylabel(f"Residual")
            plt.title(f"T(f,M,m,d) Residuals vs {name}")
            plt.tight_layout()
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()
