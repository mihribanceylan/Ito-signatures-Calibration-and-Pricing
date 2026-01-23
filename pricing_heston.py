import numpy as np
import esig, ast
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ito_utils import get_keys_and_tuples, generate_ito_correction_map, ito_from_stratonovich
from pricing_utils import log_increments, realized_variance_from_logs, realized_vol_from_logs, correlation_12, covariance_12, mc_price_with_ci, plugin_price

import os

# ===============================
# 2D Heston simulator 
# ===============================

def simulate_heston_2d(
    n_paths,
    N,
    T,
    S0,
    v0,
    kappa,
    theta,
    sigma,
    rho_S,
    rho_v,
    rho_sv,
    r,
    seed,
):
    """
    For asset i=1,2:
        dS^i_t = r S^i_t dt + sqrt(v^i_t) S^i_t dW^{S,i}_t
        dv^i_t = kappa_i (theta_i - v^i_t) dt + sigma_i sqrt(max(v^i_t,0)) dW^{v,i}_t

    We *simulate in log space* X^i_t = log S^i_t:
        dX^i_t = (r - 0.5 v^i_t) dt + sqrt(v^i_t) dW^{S,i}_t
    and then return S = exp(X) (levels). 

    Correlation among [W^{S,1}, W^{v,1}, W^{S,2}, W^{v,2}]:
        Corr(W^{S,1}, W^{S,2}) = rho_S
        Corr(W^{v,1}, W^{v,2}) = rho_v
        Corr(W^{S,i}, W^{v,i}) = rho_sv[i-1]
        Others zero.
    """
    rng = np.random.default_rng(seed)
    d = 2
    dt = T / N
    # Correlation matrix R for the Brownian motions
    R = np.eye(4)
    R[0,2] = R[2,0] = rho_S
    R[1,3] = R[3,1] = rho_v
    R[0,1] = R[1,0] = rho_sv[0]
    R[2,3] = R[3,2] = rho_sv[1]
    
    L=np.linalg.cholesky(R)
    
    # Arrays for log-prices and variances
    X = np.zeros((n_paths, N+1, d), dtype=np.float64)  # log-prices
    V = np.zeros((n_paths, N+1, d), dtype=np.float64)  # variances
    # Initial values for prices and variances
    X[:,0,:] = np.log(S0)        
    V[:,0,:] = v0
    # Simulate the asset prices and volatilities
    for t in range(N):
        Z = rng.standard_normal((n_paths, 4))
        Zc = Z @ L.T # Apply the Cholesky decomposition to get the correlated random variables
        ZS1, ZV1, ZS2, ZV2 = Zc[:,0], Zc[:,1], Zc[:,2], Zc[:,3]
        Vt = np.maximum(V[:,t,:], 0.0)
        dV = (
            kappa * (theta - Vt) * dt
            + sigma * np.sqrt(Vt) * np.sqrt(dt) * np.stack([ZV1, ZV2], axis=1)
        )
        V_next = np.maximum(V[:,t,:] + dV, 0.0)
        dX1 = (r - 0.5 * Vt[:,0]) * dt + np.sqrt(Vt[:,0]) * np.sqrt(dt) * ZS1
        dX2 = (r - 0.5 * Vt[:,1]) * dt + np.sqrt(Vt[:,1]) * np.sqrt(dt) * ZS2

        X[:,t+1,0] = X[:,t,0] + dX1
        X[:,t+1,1] = X[:,t,1] + dX2
        V[:,t+1,:] = V_next

    S = np.exp(X)  
    time = np.linspace(0.0, T, N+1)
    S_time = np.stack([np.column_stack([time, S[i]]) for i in range(n_paths)], axis=0)
    return S, S_time, V

# ===============================
# Streams & augmentation
# ===============================

def build_log_time(S_time):
    """Takes LEVELS in S_time and returns (t, log S) stream."""
    n_paths, N1, d1 = S_time.shape
    X = np.log(np.maximum(S_time[:,:,1:], 1e-300))
    times = S_time[:,:,0]
    S_time_log = np.zeros_like(S_time)
    for p in range(n_paths):
        S_time_log[p] = np.column_stack((times[p], X[p]))
    return S_time_log


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

def slice_single_asset_stream(S_time_like, j):
    """
    Slices the stream data for a single asset.

    Parameters:
    - S_time_like: Time-stamped price data
    - j: Index of the asset

    Returns:
    - S_time_like sliced for the j-th asset
    """
    return S_time_like[:, :, [0, 1 + j]]


def slice_assets_stream(S_time_like, idx_list):
    """
    Slices the stream data for multiple assets.

    Parameters:
    - S_time_like: Time-stamped price data
    - idx_list: List of asset indices

    Returns:
    - S_time_like sliced for the specified assets
    """
    cols = [0] + [1 + j for j in idx_list]
    return S_time_like[:, :, cols]

# ===============================
# Payoffs: RV, Cov, Corr
# ===============================






# ===============================
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
    alpha=1e-6
    reg = Ridge(alpha=alpha, fit_intercept=False).fit(X_tr, y_tr)
    return reg, alpha

# ===============================
# Plotting
# ===============================

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
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0*ncols, 3.2*nrows), squeeze=False)
    for i, t in enumerate(tasks):
        ax = axes[i // ncols, i % ncols]
        sub = df_price[df_price["Task"]==t].set_index("Method").loc[methods]
        y = sub["Plugin"].values; x = np.arange(len(methods))
        ax.plot(x, y, marker="o",linestyle='',markersize=10)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_title(t); ax.grid(axis="y", alpha=0.25)
        tp = float(sub["TruePrice"].iloc[0]); hw = float(sub["TrueHW95"].iloc[0])
        ax.axhline(tp, color="k", linestyle="--", linewidth=1.1)
        ax.fill_between([-0.3, len(methods)-0.7], tp-hw, tp+hw,
                        color="k", alpha=0.06, zorder=0)
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


def bootstrap_mse_distributions(models, X_by_task_method, tasks, test_idx, B=300, seed=202):
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
        for method, X_full in X_by_task_method[task_name].items():
            model = models[method][task_name]
            preds_te = model.predict(X_full[test_idx])
            for _ in range(B):
                bs = rng.integers(0, n, size=n)
                mse = mean_squared_error(y_te[bs], preds_te[bs])
                rows.append({"Task": task_name, "Method": method, "MSE": mse})
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
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0*ncols, 3.4*nrows), squeeze=False)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for i, t in enumerate(tasks):
        ax = axes[i // ncols, i % ncols]
        data = [df_boot[(df_boot["Task"]==t) & (df_boot["Method"]==m)]["MSE"].values
                for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True)
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c); box.set_alpha(0.65)
        ax.set_title(t); ax.set_ylabel("MSE")
        ax.grid(axis="y", alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4,4))
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(fname, dpi=150); plt.close(fig)

# ===============================
# MAIN 
# ===============================
if __name__ == "__main__":
    os.makedirs("plots_heston_pricing", exist_ok=True)

    # --- Heston params (2 assets) ---
    n_paths = 20_000
    N = 252; T = 1.0; r = 0.0

    S0    = np.array([100.0, 80.0])
    v0    = np.array([0.04, 0.09])
    kappa = np.array([2.0, 1.8])
    theta = np.array([0.04, 0.09])
    sigma = np.array([0.50,0.60])

    rho_S  = 0.30
    rho_v  = 0.50
    rho_sv = (-0.6, -0.5)

    level = 2
    methods=["Itô","Stratonovich"]


    # --- simulate training/test set ---
    S, S_time, V = simulate_heston_2d(
        n_paths, N, T, S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma,
        rho_S=rho_S, rho_v=rho_v, rho_sv=rho_sv, r=r, seed=42
    )

    
    S_time_log_all = build_log_time(S_time)
    # --- split train/test ---
    idx_all = np.arange(n_paths)
    train_idx, test_idx = train_test_split(idx_all, test_size=0.25, random_state=1)
    # ----- realized functionals & strikes -----
    RVar = realized_variance_from_logs(S_time_log_all)  # (n_paths, 2)
    RV   = np.sqrt(RVar)
    RCov = covariance_12(slice_assets_stream(S_time_log_all, [0,1]))  # (n_paths,)
    RCorr= correlation_12(slice_assets_stream(S_time_log_all, [0,1])) # (n_paths,)

    # Strikes (use sample means/medians for illustration)
    K_rv_swap = RVar[train_idx].mean(axis=0)
    K_rv_call = RV[train_idx].mean(axis=0) * np.array([1.05, 0.95])
    K_cov_swap = float(RCov[train_idx].mean())
    K_cov_call = float(np.median(RCov[train_idx]))
    K_corr_swap= float(RCorr[train_idx].mean())
    K_corr_call= float(np.median(RCorr[train_idx]))

    RVswap = RVar - K_rv_swap
    RVcall = np.maximum(RV - K_rv_call, 0.0)
    CovSwap = RCov - K_cov_swap
    CovCall = np.maximum(RCov - K_cov_call, 0.0)
    CorrSwap= RCorr - K_corr_swap
    CorrCall= np.maximum(RCorr - K_corr_call, 0.0)

    tasks = {}
    for j in range(2):
        tasks[f"RVswap_{j+1}"] = RVswap[:,j]
        tasks[f"RVcall_{j+1}"] = RVcall[:,j]
    tasks["CovSwap"]  = CovSwap
    tasks["CovCall"]  = CovCall
    tasks["CorrSwap"] = CorrSwap
    tasks["CorrCall"] = CorrCall

    task_names = list(tasks.keys())

    # ----- features -----
    def features_for_task_streams(task_name, S_time_log_all):
        if task_name.startswith("RV"):
            j = int(task_name.split("_")[1]) - 1
            return slice_single_asset_stream(S_time_log_all, j)
        return slice_assets_stream(S_time_log_all, [0,1])

    def build_feature_family(S_time_like, level):
        n_paths, _, d1 = S_time_like.shape
        d_loc = d1 - 1
        S_aug   = build_augmented_from(S_time_like)
        ito = np.array([ito_from_stratonovich(S_aug[i], d_loc, level)[0]
                        for i in range(n_paths)])
        strato = np.array([esig.stream2sig(np.ascontiguousarray(S_time_like[i],
                            dtype=np.float64), level) for i in range(n_paths)])
        return {"Itô": ito, "Stratonovich": strato}               
        

    feats_by_task = {name: build_feature_family(
                        features_for_task_streams(name, S_time_log_all), level)
                     for name in task_names}

    

    # ----- fit models -----
    models = {m: {} for m in methods}
    mse_rows = []
    for name in task_names:
        y = tasks[name]
        y_tr, y_te = y[train_idx], y[test_idx]
        feat_family = feats_by_task[name]
        for m in methods:
            X = feat_family[m]
            reg, alpha = fit_ridge(X[train_idx], y_tr)
            yhat_tr = reg.predict(X[train_idx])
            yhat_te = reg.predict(X[test_idx])
            mse_rows.append({"Task": name, "Method": m,
                             "MSE_train": mean_squared_error(y_tr, yhat_tr),
                             "MSE_test":  mean_squared_error(y_te, yhat_te)
                             })

            models[m][name] = reg
    #df_mse = pd.DataFrame(mse_rows)
    

    # ===============================
    #  Pricing + MC 
    # ===============================

    n_mc = 25_000
    S_mc, S_time_mc, V_mc = simulate_heston_2d(
        n_mc, N, T, S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=sigma,
        rho_S=rho_S, rho_v=rho_v, rho_sv=rho_sv, r=r, seed=777
    )
    S_time_log_mc_all = build_log_time(S_time_mc)

    true_vals = {}
    RVar_mc = realized_variance_from_logs(S_time_log_mc_all)
    RV_mc   = np.sqrt(RVar_mc)
    RCov_mc = covariance_12(slice_assets_stream(S_time_log_mc_all, [0,1]))
    RCorr_mc= correlation_12(slice_assets_stream(S_time_log_mc_all, [0,1]))

    RVswap_mc = RVar_mc - K_rv_swap
    RVcall_mc = np.maximum(RV_mc - K_rv_call, 0.0)
    CovSwap_mc= RCov_mc - K_cov_swap
    CovCall_mc= np.maximum(RCov_mc - K_cov_call, 0.0)
    CorrSwap_mc= RCorr_mc - K_corr_swap
    CorrCall_mc= np.maximum(RCorr_mc - K_corr_call, 0.0)

    for j in range(2):
        true_vals[f"RVswap_{j+1}"] = RVswap_mc[:,j]
        true_vals[f"RVcall_{j+1}"] = RVcall_mc[:,j]
    true_vals["CovSwap"]  = CovSwap_mc
    true_vals["CovCall"]  = CovCall_mc
    true_vals["CorrSwap"] = CorrSwap_mc
    true_vals["CorrCall"] = CorrCall_mc

    true_price = {}; true_hw95 = {}
    for k, v in true_vals.items():
        m, hw = mc_price_with_ci(v)
        true_price[k] = float(m[0]); true_hw95[k] = float(hw[0])

    feats_mc_by_task = {}
    for name in task_names:
        stream_mc = features_for_task_streams(name, S_time_log_mc_all)
        feats_mc_by_task[name] = build_feature_family(stream_mc, level)

    price_rows = []
    for name in task_names:
        tp, hw = true_price[name], true_hw95[name]
        for m in methods:
            plug_in = plugin_price(models[m][name], feats_mc_by_task[name][m])
            price_rows.append({"Task": name, "Method": m,
                               "TruePrice": tp, "TrueHW95": hw,
                               "Plugin": plug_in, "Bias": plug_in - tp})
    df_price = pd.DataFrame(price_rows)

    plot_prices_vs_true_lines(df_price, task_names, methods,
                              fname="plots_heston_pricing/prices_vs_true_lines.png"
                              )

    # ===============================
    #  MSE bootstrap
    # ===============================
    X_by_task_method = {name: feats_by_task[name] for name in task_names}
    df_mse_boot = bootstrap_mse_distributions(models, X_by_task_method,
                                              tasks, test_idx, B=300, seed=202)
    plot_mse_boxplots_per_payoff(df_mse_boot, task_names, methods,
                                 fname="plots_heston_pricing/mse_boxplots_per_payoff.png"
                            )

    print("Saved plots in ./plots_heston_pricing/:")
    print("  - prices_vs_true_lines.png")
    print("  - mse_boxplots_per_payoff.png")
