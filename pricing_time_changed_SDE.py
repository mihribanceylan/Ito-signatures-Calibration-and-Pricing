

import os, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ito_utils import get_keys_and_tuples, generate_ito_correction_map, ito_from_stratonovich
from pricing_utils import (
    log_increments, realized_variance_from_logs, realized_vol_from_logs,
    correlation_12, covariance_12,
    rv_call_from_logs, rv_swap_from_logs,
    corr_swap_payoff, corr_call_payoff,
    cov_swap_payoff, cov_call_payoff,
    mc_price_with_ci, plugin_price,
    slice_single_asset_stream,slice_assets_stream )
import esig


# ==============================
# Cantor clock 
# ==============================
def cantor_function_grid(n_points: int, levels: int = 20, enforce_monotone: bool = True) -> np.ndarray:
    """
    Approximates the Cantor function C(t) on a uniform grid in [0,1].
    This function simulates the Cantor function as a discrete process and uses a recursive 
    method to compute its values.
    
    Parameters:
    - n_points: Number of grid points
    - levels: Number of recursive levels for the Cantor function approximation
    - enforce_monotone: Boolean indicating whether the function should be monotonically increasing

    Returns:
    - C: An array of the Cantor function values on the grid.
    """
    # Initialize time points and Cantor function array
    t = np.linspace(0.0, 1.0, n_points)
    C = np.zeros_like(t, dtype=float)
    x = t.copy()
    active = np.ones_like(t, dtype=bool)
    w = 0.5
     # Recursive calculation of the Cantor function
    for _ in range(levels):
        x *= 3.0
        digit = np.floor(x + 1e-12).astype(int)  # {0,1,2}
        x -= digit
        C += (active & (digit == 2)) * w
        active &= (digit != 1)
        x[~active] = 0.0
        w *= 0.5
    # Enforce monotonicity if required
    if enforce_monotone:
        C = np.maximum.accumulate(C)
    return np.clip(C, 0.0, 1.0)

# ==============================
# 2D time-changed SDE 
# ==============================
def simulate_cantor_price_sde_2d(n_paths, N, S0, rho, levels, seed):
                               
    """
    Simulates a 2-dimensional price SDE on Cantor time with correlated assets.

    Parameters:
    - n_paths: Number of paths to simulate
    - N: Number of time steps
    - S0: Initial prices for the two assets
    - rho: Correlation between the assets
    - levels: Number of levels for the Cantor clock
    - sigma_fn: Callable function for volatility
    - seed: Random seed for reproducibility

    Returns:
    - S: Simulated prices of shape (n_paths, N+1, 2)
    - S_time: Time-stamped prices
    - S_time_log: Time-stamped log-prices
    - C: Cantor clock used for time-stepping
    """
    
    rng = np.random.default_rng(seed)

    # time grid & Cantor clock
    t = np.linspace(0.0, 1.0, N+1)
    C = cantor_function_grid(N+1, levels=levels, enforce_monotone=True)
    dC = np.diff(C)  

    d = 2
    S0 = np.asarray(S0, dtype=float)
    
    nu = np.array([0.20, 0.30], dtype=float)
    def sigma_fn(Z): return nu * Z

    # Cholesky decomposition to introduce correlation
    L = np.linalg.cholesky(np.array([[1.0, rho],[rho, 1.0]], dtype=float))
    
    S = np.zeros((n_paths, N+1, d), dtype=float)
    S[:, 0, :] = S0
    # Simulate prices using the Cantor clock
    for k in range(N):
        Z  = rng.standard_normal((n_paths, d)) @ L.T
        dW = np.sqrt(dC[k]) * Z
        S[:, k+1, :] = S[:, k, :] + sigma_fn(S[:, k, :]) * dW
        S[:, k+1, :] = np.maximum(S[:, k+1, :], 1e-12)

    S_time = np.stack([np.column_stack([t, S[i]]) for i in range(n_paths)], axis=0)
    X = np.log(S)
    S_time_log = np.stack([np.column_stack([t, X[i]]) for i in range(n_paths)], axis=0)
    return S, S_time, S_time_log, C


# ==============================
# Streams & augmentation on PRICE
# ==============================


def build_augmented_from(S_time_like):
    """
    Augments the time-stamped price data by adding quadratic variation features.

    Parameters:
    - S_time_like: Time-stamped price data

    Returns:
    - S_aug: Augmented data with time, log prices, and quadratic variations
    """
    n_paths, N1, d1 = S_time_like.shape
    d_loc = d1 - 1
    X = S_time_like[:,:,1:]
    triu = np.triu_indices(d_loc)
    qv_dim = len(triu[0])
    QV_reduced = np.zeros((n_paths, N1, qv_dim), dtype=np.float64)
    for p in range(n_paths):
        cum = np.zeros((d_loc, d_loc), dtype=np.float64)
        for t in range(1, N1):
            incr = X[p,t] - X[p,t-1]
            cum += np.outer(incr, incr)
            QV_reduced[p,t] = cum[triu]
    S_aug = np.zeros((n_paths, N1, 1 + d_loc + qv_dim), dtype=np.float64)
    for p in range(n_paths):
        for t in range(N1):
            S_aug[p,t] = np.hstack((S_time_like[p,t,0], X[p,t], QV_reduced[p,t]))
    return S_aug

# ==============================
# Ridge 
# ==============================

def fit_ridge(X_tr, y_tr):
    """
    Fits Ridge regression.

    Parameters:
    - X_tr: Training features
    - y_tr: Training labels

    Returns:
    - reg: The fitted Ridge regression model
    """
    alpha = 1e-6
    reg = Ridge(alpha=alpha, fit_intercept=False).fit(X_tr, y_tr)
    return reg, alpha

# ==============================
# Feature builders 
# ==============================
def build_feature_family(S_time_like, level):
    """
    Builds a family of features using the PRICE signatures for the given paths.

    Parameters:
    - S_time_like: Time-stamped price data
    - level: Depth of the signature

    Returns:
    - Dictionary containing the Stratonovich and Itô signatures
    """
    n_paths, _, d1 = S_time_like.shape
    d_loc = d1 - 1
    S_aug   = build_augmented_from(S_time_like)
    ito = np.array([ito_from_stratonovich(S_aug[i], d_loc, level)[0] for i in range(n_paths)])
    strato = np.array([esig.stream2sig(np.ascontiguousarray(S_time_like[i], dtype=np.float64), level)
                       for i in range(n_paths)])
    return {"Itô": ito, "Stratonovich": strato}

# ==============================
# Plot helpers
# ==============================
def plot_prices_vs_true_lines(df_price, tasks, methods, fname):
    """
    Plots the predicted prices versus the true prices with confidence intervals.

    Parameters:
    - df_price: DataFrame containing the predicted prices and true values
    - tasks: List of task names
    - methods: List of methods used for pricing (Itô, Stratonovich)
    - fname: Filename to save the plot
    """
    n = len(tasks); ncols = 3; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.2*nrows), squeeze=False)
    for i, t in enumerate(tasks):
        ax = axes[i // ncols, i % ncols]
        sub = df_price[df_price["Task"]==t].set_index("Method").loc[methods]
        y = sub["Plugin"].values; x = np.arange(len(methods))
        ax.plot(x, y, marker="o",linestyle='', markersize=10)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_title(t); ax.grid(axis="y", alpha=0.25)
        tp = float(sub["TruePrice"].iloc[0]); hw = float(sub["TrueHW95"].iloc[0])
        ax.axhline(tp, color="k", linestyle="--", linewidth=1.1)
        ax.fill_between([-0.3, len(methods)-0.7], tp-hw, tp+hw, color="k", alpha=0.06, zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3,3))
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    ax0 = axes[0, 0]
    handles = [
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.1, label="MC price"),
        Patch(facecolor="k", alpha=0.06, edgecolor="none", label="95% MC CI")
              ]
    ax0.legend(handles=handles, loc="upper left", frameon=False, handlelength=2.0)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(fname, dpi=150); plt.close(fig)

def bootstrap_mse_distributions(models, feats_by_task, tasks, test_idx, methods, B=300, seed=202):
    """
    Bootstrap procedure to compute the MSE distribution for each method and task.

    Parameters:
    - models: Dictionary of trained models
    - feats_by_task: Features for each task
    - tasks: Dictionary of tasks and their true values
    - test_idx: Test indices for bootstrapping
    - methods: List of methods used
    - B: Number of bootstrap samples
    - seed: Random seed for reproducibility

    Returns:
    - DataFrame containing the bootstrapped MSE values
    """
    rng = np.random.default_rng(seed)
    rows = []
    for task_name, y_all in tasks.items():
        y_te = y_all[test_idx]
        n = len(y_te)
        for method in methods:
            X_full = feats_by_task[task_name][method]
            model = models[method][task_name]
            preds_te = model.predict(X_full[test_idx])
            for _ in range(B):
                bs = rng.integers(0, n, size=n)   # resample test indices
                mse = mean_squared_error(y_te[bs], preds_te[bs])
                rows.append({"Task": task_name, "Method": method, "MSE": float(mse)})
    return pd.DataFrame(rows)

def plot_mse_boxplots_per_payoff(df_boot, tasks, methods, fname):
    """
    Plots the MSE distributions using boxplots for each task and method.

    Parameters:
    - df_boot: DataFrame containing the bootstrapped MSE values
    - tasks: List of task names
    - methods: List of methods used
    - fname: Filename to save the plot
    """
    n = len(tasks); ncols = 3; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.4*nrows), squeeze=False)
    for i, t in enumerate(tasks):
        ax = axes[i // ncols, i % ncols]
        data = [df_boot[(df_boot["Task"]==t) & (df_boot["Method"]==m)]["MSE"].values
                for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True)
        for box in bp["boxes"]:
            box.set_alpha(0.65)
        ax.set_title(t); ax.set_ylabel("MSE")
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4,4))
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(fname, dpi=150); plt.close(fig)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    outdir = "plots_pricing_time_changed_SDE"
    os.makedirs(outdir, exist_ok=True)

    # Simulation parameters
    n_paths = 20000
    N = 252
    levels_cantor = 18
    rho = 0.5
    S0 = np.array([100.0, 80.0])
    level_sig = 2
    methods = ["Itô", "Stratonovich"]

    # Simulate TRAIN/TEST set
    idx_all = np.arange(n_paths)
    train_idx, test_idx = train_test_split(idx_all, test_size=0.25, random_state=1)
    S, S_time, S_time_log_all, C = simulate_cantor_price_sde_2d(
    n_paths, N, S0=S0, rho=rho, levels=levels_cantor, seed=7)
    
                     


    # Compute payoffs
    RV_all   = realized_vol_from_logs(S_time_log_all)         # (n_paths, 2)
    RVar_all = realized_variance_from_logs(S_time_log_all)    # (n_paths, 2)
    Corr_all = correlation_12(S_time_log_all)                 # (n_paths,)
    Cov_all  = covariance_12(S_time_log_all)              # (n_paths,)
    # Split data for training and testing
    RV_tr, RVar_tr = RV_all[train_idx], RVar_all[train_idx]
    Corr_tr, Cov_tr = Corr_all[train_idx], Cov_all[train_idx]
    # Define strike prices for different payoffs
    K_rv_call = RV_tr.mean(axis=0) * np.array([1.05, 0.95])
    K_rv_swap = RVar_tr.mean(axis=0)
    K_corr_swap = float(np.mean(Corr_tr))
    K_corr_call = float(np.median(Corr_tr))
    K_cov_swap  = float(np.mean(Cov_tr))
    K_cov_call  = float(np.median(Cov_tr))

    tasks = {}
    for j in range(2):
        tasks[f"RVcall_{j+1}"] = rv_call_from_logs(slice_single_asset_stream(S_time_log_all, j), K_rv_call[j])
        tasks[f"RVswap_{j+1}"] = rv_swap_from_logs (slice_single_asset_stream(S_time_log_all, j), K_rv_swap[j])
    tasks["CorrSwap"] = corr_swap_payoff(slice_assets_stream(S_time_log_all, [0,1]), K_corr_swap)
    tasks["CorrCall"] = corr_call_payoff(slice_assets_stream(S_time_log_all, [0,1]), K_corr_call)
    tasks["CovSwap"]  = cov_swap_payoff (slice_assets_stream(S_time_log_all, [0,1]), K_cov_swap)
    tasks["CovCall"]  = cov_call_payoff (slice_assets_stream(S_time_log_all, [0,1]), K_cov_call)

    task_names = list(tasks.keys())

    
    def features_for_task_streams(task_name, S_time_log_all):
        if task_name.startswith(("RVcall_", "RVswap_")):
            j = int(task_name.split("_")[1]) - 1
            return slice_single_asset_stream(S_time_log_all, j)
        else:
            return slice_assets_stream(S_time_log_all, [0,1])

    # Choose features for each task and fit models
    feats_by_task = {
        name: build_feature_family(features_for_task_streams(name, S_time_log_all), level_sig)
        for name in task_names
    }

    

     # Fit Ridge regression models and calculate MSE
    models = {m: {} for m in methods}
    mse_rows = []
    for name in task_names:
        y = tasks[name]
        y_tr, y_te = y[train_idx], y[test_idx]
        feats = feats_by_task[name]
        for m in methods:
            X = feats[m]
            reg, alpha = fit_ridge(X[train_idx], y_tr)
            models[m][name] = reg
            yhat_tr = reg.predict(X[train_idx])
            yhat_te = reg.predict(X[test_idx])
            mse_rows.append({"Task": name, "Method": m,
                             "MSE_train": mean_squared_error(y_tr, yhat_tr),
                             "MSE_test":  mean_squared_error(y_te, yhat_te)
                             })
    df_mse = pd.DataFrame(mse_rows)
    

    
    # ---------- PRICING COMPARISON  ----------
    n_mc = 25_000
    S_mc, S_time_mc, S_time_log_mc_all, C_mc = simulate_cantor_price_sde_2d(
                 n_paths=n_mc, N=N, S0=S0, rho=rho, levels=levels_cantor, seed=777
                   )

    # Compute true payoffs using MC paths 
    true_vals = {}
    for j in range(2):
       s1 = slice_single_asset_stream(S_time_log_mc_all, j)
       true_vals[f"RVcall_{j+1}"] = rv_call_from_logs(s1, K_rv_call[j])
       true_vals[f"RVswap_{j+1}"] = rv_swap_from_logs (s1, K_rv_swap[j])
    s2 = slice_assets_stream(S_time_log_mc_all, [0, 1])
    true_vals["CorrSwap"] = corr_swap_payoff(s2, K_corr_swap)
    true_vals["CorrCall"] = corr_call_payoff(s2, K_corr_call)
    true_vals["CovSwap"]  = cov_swap_payoff (s2, K_cov_swap)
    true_vals["CovCall"]  = cov_call_payoff (s2, K_cov_call)

    # Build MC features 
    feats_mc_by_task = {
           name: build_feature_family(features_for_task_streams(name, S_time_log_mc_all), level_sig)
           for name in task_names
          }

    # Compute prices
    rows = []
    for name in task_names:
        tp, hw = mc_price_with_ci(true_vals[name])
        for m in methods:
             plug_in= plugin_price(models[m][name], feats_mc_by_task[name][m])
             rows.append({
                    "Task": name, "Method": m,
                    "TruePrice": float(tp[0]), "TrueHW95": float(hw[0]),
                    "Plugin": plug_in,
                    "Bias": plug_in - float(tp[0])
                        })
    df_price = pd.DataFrame(rows)
    df_price.to_csv(os.path.join(outdir, "price_table.csv"), index=False)

    # Plot prices vs. true lines
    plot_prices_vs_true_lines(
         df_price,
         tasks=task_names,
         methods=methods,
         fname=os.path.join(outdir, "prices_vs_true_lines.png")
             )

    # ---------- Test-MSE boxplots  ----------
    df_boot = bootstrap_mse_distributions(models, feats_by_task, tasks, test_idx,
                                      methods, B=300, seed=202)
    plot_mse_boxplots_per_payoff(
        df_boot, task_names, methods,
        fname=os.path.join(outdir, "mse_boxplots_per_payoff.png")
    )

    print(f"\nSaved outputs in ./{outdir}:")
    print("  - price_table.csv")
    print("  - prices_vs_true_lines.png")
    print("  - mse_boxplots_per_payoff.png")
    print("\nHead of MSE table:")
    print(df_mse)
