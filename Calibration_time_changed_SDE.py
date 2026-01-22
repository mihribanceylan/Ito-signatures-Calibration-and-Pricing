import os, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import iisignature   # pip install iisignature
import esig          # pip install esig

# -----------------------------
# Cantor clock C(t) 
# -----------------------------
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
        digit = np.floor(x + 1e-12).astype(int)
        x -= digit

        C += (active & (digit == 2)) * w
        active &= (digit != 1)
        x[~active] = 0.0 
        w *= 0.5
    # Enforce monotonicity if required
    if enforce_monotone:
        C = np.maximum.accumulate(C)

    return np.clip(C, 0.0, 1.0)

# -----------------------------
# Plot: Cantor function
# -----------------------------
def plot_cantor(outdir="calib_outputs_cantor_general",
                n_points=2001, levels=20, enforce_monotone=True):
    """
    Plots the Cantor function on a grid. Two plots are created:
    1. The entire Cantor function
    2. A zoomed-in view of the middle third [1/3, 2/3]
    
    Parameters:
    - outdir: Directory to save the plots
    - n_points: Number of grid points
    - levels: Number of recursive levels for the Cantor function approximation
    - enforce_monotone: Boolean indicating whether the function should be monotonically increasing
    """
    os.makedirs(outdir, exist_ok=True)
    # Generate the Cantor function
    t = np.linspace(0.0, 1.0, n_points)
    C = cantor_function_grid(n_points=n_points, levels=levels,
                             enforce_monotone=enforce_monotone)
    # Calculate the differences of the Cantor function
    diffs = np.diff(C)
    print(f"[Cantor] min ΔC = {diffs.min():.3e}, max ΔC = {diffs.max():.3e}")
    # Plot the full Cantor function
    plt.figure(figsize=(10,4))
    plt.step(t, C, where="post", linewidth=1.6)
    plt.xlabel("t"); plt.ylabel("C(t)")
    plt.title(f"Cantor function (levels={levels}, n={n_points})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cantor_full.png"), dpi=150)
    plt.close()
    # Zoom plot for the middle third [1/3, 2/3]
    mask = (t >= 1/3) & (t <= 2/3)
    plt.figure(figsize=(10,4))
    plt.step(t[mask], C[mask], where="post", linewidth=1.6)
    plt.xlabel("t"); plt.ylabel("C(t)")
    plt.title("Cantor function — zoom on middle third [1/3, 2/3]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cantor_zoom_middle_third.png"), dpi=150)
    plt.close()

# -----------------------------
# User-definable drift/diffusion
# -----------------------------
def mu_fn(t, x):
    a0, a1 = 0.0, 0.0
    return a0 + a1 * x

def sigma_fn(t, x):
    s0, s1 = 1.0, 0.3
    return s0 * (1.0 + s1 * np.tanh(x))

# -----------------------------
# Simulate W_{C(t)} and X_t
# -----------------------------
def simulate_SDE_cantor(n_points: int, levels: int, rng, t_final: float = 1.0, mu=mu_fn, sigma=sigma_fn):
    """
    Simulates the time-changed SDE with Cantor clock over the interval [0, t_final].
    This function simulates both the Cantor clock and the underlying stochastic process X_t.

    Parameters:
    - n_points: Number of grid points
    - levels: Number of recursive levels for the Cantor clock
    - rng: Random number generator for the simulation
    - t_final: End time for the simulation interval

    Returns:
    - t: Time points
    - C: Cantor function values
    - W: time-changed Brownian motion values (W_{C(t)})
    - X: Solution of the stochastic process X_t
    """
    # Normalized time grid for [0,1]
    t_norm = np.linspace(0.0, 1.0, n_points)
    C = cantor_function_grid(n_points, levels=levels)

    t = t_final * t_norm
     # Simulate W_{C(t)}
    W = np.zeros_like(t)
    for k in range(n_points - 1):
        dC = C[k+1] - C[k]
        dW = rng.normal(0.0, np.sqrt(dC)) if dC > 0.0 else 0.0
        W[k+1] = W[k] + dW
    # Simulate the stochastic process X_t
    X = np.zeros_like(t)
    dt = t_final / (n_points - 1)
    for k in range(n_points - 1):
        mx = mu(t[k], X[k]) * dt
        sx = sigma(t[k], X[k]) * (W[k+1] - W[k])  # dW_C
        X[k+1] = X[k] + mx + sx

    return t, C, W, X

# -----------------------------
# Streams 
# -----------------------------
def stream_stratonovich(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    n = len(t)
    t_scaled = np.arange(n, dtype=float) / max(n, 1)
    return np.column_stack([t_scaled, W])  

def stream_ito_augmented(t: np.ndarray, W: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = len(t)
    t_scaled = np.arange(n, dtype=float) / max(n, 1)
    return np.column_stack([t_scaled, W, C])  

# -----------------------------
# Stratonovich prefix signatures
# -----------------------------
def prefix_stratonovich_df(stream: np.ndarray, depth: int) -> pd.DataFrame:
    """
    Calculates the Stratonovich prefix signatures for a given stream and depth.
    
    Parameters:
    - stream: The input stream of the process [t, W].
    - depth: The depth of the signature to compute.

    Returns:
    - A DataFrame containing the Stratonovich prefix signatures.
    """
    d = stream.shape[1]
    # Get the signature keys for the given depth
    keys_raw = esig.sigkeys(d, depth).strip().split()
    keys = keys_raw if (keys_raw and keys_raw[0] == "()") else ["()"] + keys_raw
    rows = []
    for i in range(1, len(stream) + 1):
        # Compute the signature for the stream up to index i
        S = stream[:i, :]
        v = iisignature.sig(S, depth)
        # Append a row with the signature and the prefix (starting with 1.0, corresponding to the empty index)
        rows.append(np.concatenate(([1.0], v)))
        # Return the result as a DataFrame with the corresponding keys
    return pd.DataFrame(rows, columns=keys)

# -----------------------------
# Itô correction map
# -----------------------------
def get_keys_and_tuples(d_aug, level):
    """
    Converts the signature keys into a list of tuples.
    
    Parameters:
    - d_aug: The augmented dimension of the process (W, C).
    - level: The depth of the signature.
    
    Returns:
    - A list of tuples representing the keys.
    """
    keys_str = esig.sigkeys(d_aug, level).split()
    keys_tuple = []
    
    for k in keys_str:
        # Convert the string representation of keys into tuples
        val = ast.literal_eval(k)
        if isinstance(val, int):
            val = (val,)
        elif not isinstance(val, tuple):
            raise TypeError(f"Unexpected key: {k}")
        keys_tuple.append(val)
    return keys_tuple

def generate_ito_correction_map(d_aug, d_loc, level):
    """
    Generates the Itô correction map for the given depth and dimensions.
    
    Parameters:
    - d_aug: The augmented dimension of the process.
    - d_loc: The dimension of the local process.
    - level: The depth of the signature.
    
    Returns:
    - keys: The list of keys for the signature.
    - cmap: A dictionary containing the correction map.
    """
    # Get the keys for the signature
    keys = get_keys_and_tuples(d_aug, level)
    # Generate the QV pairs for correction
    qv_pairs = [(a,b) for a in range(2, d_loc+2) for b in range(a, d_loc+2)]
    def map_to_qv_channel(i, j):
        """
        Maps indices i and j to a QV channel.
        """
        pair = (min(i,j), max(i,j))
        return (2 + d_loc) + qv_pairs.index(pair)

    def non_adjacent_subsets(idxs):
        """
        Generates non-adjacent subsets of the indices.
        """
        res = [[]]
        for idx in idxs:
            res += [s+[idx] for s in res if (not s) or (idx - s[-1] > 1)]
        return res[1:]

    cmap = {}
    for key in keys:
        if key == ():
            cmap[key] = []
            continue
        # Process each key and create the correction map
        idxs = tuple(int(x) for x in key)
        cand = [j for j in range(len(idxs)-1)
                if 2 <= idxs[j] <= d_loc+1 and 2 <= idxs[j+1] <= d_loc+1]
        combinations = []
        for subset in non_adjacent_subsets(cand):
            new_index = []
            skip = False
            for j in range(len(idxs)):
                if skip:
                    skip = False
                    continue
                if j in subset:
                    new_index.append(map_to_qv_channel(idxs[j], idxs[j+1]))
                    skip = True
                else:
                    new_index.append(idxs[j])
            combinations.append((tuple(new_index), (-0.5)**len(subset)))
        cmap[key] = combinations
    return keys, cmap

def parse_key_to_word(key_str):
    """
    Converts a key string to a tuple format.
    
    Example: '(2,3)' -> (2,3), '(2)' -> (2,), '()' -> ()
    """
    if key_str == "()":
        return ()
    obj = ast.literal_eval(key_str)
    if isinstance(obj, int):
        return (obj,)
    return tuple(obj)

def select_ito_features_ending_in_WC(ito_df):
    """
    Selects the Itô signature features that end with the index 2 (corresponding to W_C).
    
    Parameters:
    - ito_df: DataFrame containing the Itô signature features.
    
    Returns:
    - A filtered DataFrame containing only features that end in W_C.
    """
    selected_cols = []
    for col in ito_df.columns:
        if col == "()":
            continue
        word = parse_key_to_word(col)
        if len(word) > 0 and word[-1] == 2:  # last index= W_C
            selected_cols.append(col)
    return ito_df[selected_cols], selected_cols

def prefix_ito_df(stream_aug: np.ndarray, d_loc: int, depth: int) -> pd.DataFrame:
    """
    Computes the Itô prefix signatures with correction terms for a given stream and depth.
    
    Parameters:
    - stream_aug: The augmented stream [t, W, C].
    - d_loc: Local dimension.
    - depth: The depth of the signature.
    
    Returns:
    - A DataFrame containing the Itô corrected prefix signatures.
    """
    d_aug = stream_aug.shape[1]
    keys_raw = esig.sigkeys(d_aug, depth).strip().split()
    keys = keys_raw if (keys_raw and keys_raw[0] == "()") else ["()"] + keys_raw
    keys_tuples, cmap = generate_ito_correction_map(d_aug, d_loc, depth)
    k2i = {k:i for i,k in enumerate(keys_tuples)}

    rows = []
    for i in range(1, len(stream_aug) + 1):
        S = stream_aug[:i, :]
        strat_no1 = iisignature.sig(S, depth)
        strat = np.concatenate(([1.0], strat_no1))
        ito = strat.copy()
        for idx, key in enumerate(keys_tuples):
            if len(key) < 2:
                continue
            for new_key, fac in cmap.get(key, []):
                j = k2i.get(new_key)
                if j is not None:
                    ito[idx] += fac * strat[j]
        rows.append(ito)
    return pd.DataFrame(rows, columns=keys)

# ==== Features ===================
def build_features(t, C, W, depth):
    """
    Builds the features for Stratonovich and Itô signatures.
    
    Parameters:
    - t: Time points.
    - C: Cantor function values.
    - W: time-changed Brownian motion.
    - depth: Depth of the signature.

    Returns:
    - Strat_df: DataFrame of Stratonovich features.
    - Ito_df: DataFrame of Itô features.
    """
    S_strato = stream_stratonovich(t, W)
    S_ito    = stream_ito_augmented(t, W, C)
    Strat_df = prefix_stratonovich_df(S_strato, depth)
    Ito_df   = prefix_ito_df(S_ito, d_loc=1, depth=depth)
    return Strat_df, Ito_df

# -----------------------------
# Lasso: Calibration
# -----------------------------
def fit_lasso_full_path(Xdf, y, alpha=1e-5):
    """
    Fits a Lasso regression model on the given data (Xdf, y).
    
    Parameters:
    - Xdf: DataFrame containing the features.
    - y: Target variable.
    - alpha: Regularization strength for the Lasso regression.

    Returns:
    - A dictionary containing:
        - "reg": The fitted Lasso regression model.
        - "scaler": The scaler used to standardize the features.
        - "yhat_train": The predicted values on the training data.
        - "mse_train": The mean squared error of the predictions on the training data.
    """
    # Convert input DataFrame to numpy array
    X = Xdf.values
    # Standardize the features (mean=0, std=1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_s = scaler.fit_transform(X)
    # Fit a Lasso regression model
    reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    reg.fit(X_s, y)
    # Predict values using the trained model
    yhat = reg.predict(X_s)
    # Compute the mean squared error on the training data
    mse_train = mean_squared_error(y, yhat)

    return {
        "reg": reg,
        "scaler": scaler,
        "yhat_train": yhat,
        "mse_train": mse_train
    }

def apply_lasso_model(Xdf, y, reg, scaler):
    """
    Applies the trained Lasso regression model to new data (Xdf) to make predictions.
    
    Parameters:
    - Xdf: DataFrame containing the features.
    - y: Target variable.
    - reg: The trained Lasso regression model.
    - scaler: The scaler used to standardize the features before fitting.

    Returns:
    - mse: Mean squared error of the predictions on the test data.
    - yhat: Predicted values on the test data.
    """
    # Convert input DataFrame to numpy array
    X = Xdf.values
    # Standardize the features using the same scaler
    X_s = scaler.transform(X)
    # Make predictions using the fitted model
    yhat = reg.predict(X_s)
    # Compute the mean squared error on the test data
    mse = mean_squared_error(y, yhat)
    return mse, yhat

# -----------------------------
# Calibration on a Single Path (T_train = 1.0)
# -----------------------------
def calibrate_on_single_path(rng, n_points, levels, depth, alpha,
                             outdir: str):
    """
    Performs calibration using Lasso regression on a single training path generated from a time-changed SDE with Cantor-clock.
    
    Parameters:
    - rng: Random number generator.
    - n_points: Number of points for discretization in the training path.
    - levels: Number of levels in the Cantor process.
    - depth: Depth of the signature features to extract.
    - alpha: Regularization strength for Lasso regression.
    - use_state_dependent: Boolean flag to toggle between state-dependent or constant drift/diffusion.
    - outdir: Output directory to save results.
    
    Returns:
    - A dictionary containing the training path, the regression results, and the errors.
    """
    
    # Simulate the time-changed SDE
    t_tr, C_tr, W_tr, X_tr = simulate_SDE_cantor(
        n_points=n_points, levels=levels, rng=rng, t_final=1.0
    )

    # Build the Stratonovich and Itô signature features
    Strat_df_tr, Ito_df_tr = build_features(t_tr, C_tr, W_tr, depth)

    # Select Itô features that end with W_C
    Ito_B_df_tr, ito_B_keys_tr = select_ito_features_ending_in_WC(Ito_df_tr)

    # Calibrate using Lasso regression for Stratonovich and Itô signatures
    res_strat = fit_lasso_full_path(Strat_df_tr, X_tr, alpha=alpha)
    res_ito = fit_lasso_full_path(Ito_B_df_tr, X_tr, alpha=alpha)
    # Restore the original drift and diffusion functions if necessary
    
    # Plot the results of the calibration path
    plt.figure(figsize=(11, 6))
    plt.plot(t_tr, X_tr, label="Cantor-type SDE (Train)", color="black", lw=1.6)
    plt.plot(t_tr, res_strat["yhat_train"], label="Stratonovich", alpha=0.9)
    plt.plot(t_tr, res_ito["yhat_train"], label="Itô", alpha=0.9, linestyle='dashed')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "train_path_fit.png"), dpi=150)
    plt.close()

    return {
        "t_train": t_tr,
        "X_train": X_tr,
        "mse_strato_train": res_strat["mse_train"],
        "mse_ito_train": res_ito["mse_train"],
        "reg_strato": res_strat["reg"],
        "scaler_strato": res_strat["scaler"],
        "reg_ito": res_ito["reg"],
        "scaler_ito": res_ito["scaler"]
    }

# -----------------------------
#  Monte-Carlo on Test Paths (T_test = 0.5)
# -----------------------------
def summarize_mc(df, col):
    """
    Summarizes the Monte Carlo results for a given column in the DataFrame (df).
    
    Parameters:
    - df: DataFrame containing the Monte Carlo results.
    - col: Column name to summarize (contains MSE results).

    Returns:
    - m: Mean value of the column.
    - sd: Standard deviation of the column.
    - ci: 95% confidence interval (lower, upper).
    """
    n = len(df)
    m  = df[col].mean()
    sd = df[col].std(ddof=1)
    se = sd / np.sqrt(max(n,1))
    ci = (m - 1.96*se, m + 1.96*se)
    return m, sd, ci

def run_test_paths(R, base_seed, n_points_train, levels, depth,
                   reg_strato, scaler_strato,
                   reg_ito,    scaler_ito,
                   outdir: str):
    """
    Simulates R test paths on the interval [0, 0.5], using the same time step as in the training. 
    Each path is used to test the performance of the Stratonovich and Itô models.
    
    Parameters:
    - R: Number of test paths to simulate.
    - base_seed: The seed for random number generation, used to generate different test paths.
    - n_points_train: Number of points for discretization in the training paths.
    - levels: Number of levels for the Cantor process.
    - depth: Depth of the signature features.
    - alpha: Regularization strength for Lasso regression.
    - use_state_dependent: Boolean flag to toggle between state-dependent or constant drift/diffusion functions.
    - reg_strato: The trained Stratonovich regression model.
    - scaler_strato: The scaler used for Stratonovich features.
    - reg_ito: The trained Itô regression model.
    - scaler_ito: The scaler used for Itô features.
    - outdir: Output directory to save results.

    Returns:
    - mc_df: DataFrame containing the test results (MSEs for Stratonovich and Itô).
    """
    

    
    

    T_test = 0.5
    dt_train = 1.0 / (n_points_train - 1)
    n_points_test = int(T_test / dt_train) + 1
    rows = []
    for r in range(R):
        # Create a new RNG for each test path
        rng_r = np.random.default_rng(base_seed + r)

        # Simulate the test path
        t_te, C_te, W_te, X_te = simulate_SDE_cantor(
            n_points=n_points_test, levels=levels, rng=rng_r, t_final=T_test
        )

        # Generate Stratonovich and Itô signature features
        Strat_df_te, Ito_df_te = build_features(t_te, C_te, W_te, depth)

        # Select the Itô features that correspond to W_C 
        Ito_B_df_te, _ = select_ito_features_ending_in_WC(Ito_df_te)
        # Restore the original drift and diffusion functions if not state-dependent
        

        # Apply the Lasso model to the Stratonovich and Itô features
        mseS_te, yhatS_te = apply_lasso_model(Strat_df_te, X_te, reg_strato, scaler_strato)
        mseI_te, yhatI_te = apply_lasso_model(Ito_B_df_te, X_te, reg_ito, scaler_ito)
        # Calculate the difference in MSE between Stratonovich and Itô models
        diff = mseS_te - mseI_te
        # Store the results for this test path
        rows.append({
            "seed_index": r,
            "mse_strato_test": mseS_te,
            "mse_ito_test": mseI_te,
            "diff_test": diff
        })
    # Save the Monte Carlo results to a CSV file
    mc_df = pd.DataFrame(rows)
    mc_df.to_csv(os.path.join(outdir, "mc_results_test_paths.csv"), index=False)
    return mc_df

#Summary of Monte-Carlo Test Results
def print_test_summary(mc_df):
    """
    Summarizes the Monte Carlo results for the out-of-sample test paths.
    This function prints the mean, standard deviation, and 95% confidence intervals for
    the MSE of both the Stratonovich and Itô models, as well as the difference between them.
    

    Parameters:
    - mc_df: DataFrame containing the Monte Carlo results (MSE for Stratonovich and Itô).
    """
    mS, sS, ciS = summarize_mc(mc_df, "mse_strato_test")
    mI, sI, ciI = summarize_mc(mc_df, "mse_ito_test")
    md, sd, cid = summarize_mc(mc_df, "diff_test")
    winrate_I = float((mc_df["diff_test"] > 0).mean())

    print("\n=== Monte-Carlo summary (OUT-OF-SAMPLE / TEST MSE, T=0.5) ===")
    print(f"Strato  mean={mS:.3e}  sd={sS:.3e}  95%CI=[{ciS[0]:.3e}, {ciS[1]:.3e}]")
    print(f"Itô     mean={mI:.3e}  sd={sI:.3e}  95%CI=[{ciI[0]:.3e}, {ciI[1]:.3e}]")
    print(f"Diff (Strato - Itô) mean={md:.3e}  sd={sd:.3e}  95%CI=[{cid[0]:.3e}, {cid[1]:.3e}]")
    print(f"Itô better out-of-sample (diff>0): {100.0*winrate_I:.1f}%")

# Plot a Test Path for a Specific Seed
def plot_test_path_for_seed(seed_idx, base_seed, n_points_train, levels, depth, alpha,
                            reg_strato, scaler_strato,
                            reg_ito,    scaler_ito,
                            outdir: str,
                            suffix: str = "random"):
    

    T_test = 0.5
    dt_train = 1.0 / (n_points_train - 1)
    n_points_test = int(T_test / dt_train) + 1

    rng_r = np.random.default_rng(base_seed + seed_idx)

    

    # Simulate test path
    t_te, C_te, W_te, X_te = simulate_SDE_cantor(
        n_points=n_points_test, levels=levels, rng=rng_r, t_final=T_test
    )
    
    # Generate Stratonovich and Itô signature features
    Strat_df_te, Ito_df_te = build_features(t_te, C_te, W_te, depth)

    # Select the Itô features that correspond to W_C
    Ito_B_df_te, _ = select_ito_features_ending_in_WC(Ito_df_te)

    

    # Apply the Lasso model for both Stratonovich and Itô features
    mseS_te, yhatS_te = apply_lasso_model(Strat_df_te, X_te, reg_strato, scaler_strato)
    mseI_te, yhatI_te = apply_lasso_model(Ito_B_df_te, X_te, reg_ito, scaler_ito)

    # Plot the results
    plt.figure(figsize=(11, 6))
    plt.plot(t_te, X_te, label="Cantor-type SDE (Test, T=0.5)", color="black", lw=1.6)
    plt.plot(t_te, yhatS_te, label="Stratonovich", alpha=0.9)
    plt.plot(t_te, yhatI_te, label="Itô", alpha=0.9, linestyle='dashed')
    plt.legend(loc='upper left')
    plt.tight_layout()
    fname = os.path.join(outdir, f"test_path_seed_{seed_idx}_{suffix}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

# -----------------------------
# MAIN:  Run the Calibration and Monte-Carlo Simulation
# -----------------------------
if __name__ == "__main__":
    outdir = "calibration_time_changed_SDE"
    os.makedirs(outdir, exist_ok=True)

    #  --- Set up parameters ---
    n_points_train = 2000   
    levels   = 12
    depth    = 2
    alpha    = 1e-5

    # Plot the Cantor function
    plot_cantor(outdir=outdir, n_points=1000, levels=levels, enforce_monotone=True)

    # ---------- Calibration on a single training path (T=1) ----------
    CALIB_SEED = 123
    rng_calib = np.random.default_rng(CALIB_SEED)
    calib_res = calibrate_on_single_path(
        rng=rng_calib,
        n_points=n_points_train,
        levels=levels,
        depth=depth,
        alpha=alpha,
        outdir=outdir
    )
    # Print results of the calibration on a single path
    print("\n=== Calibration  ===")
    print(f"Stratonovich  MSE_train={calib_res['mse_strato_train']:.6g}")
    print(f"Itô           MSE_train={calib_res['mse_ito_train']:.6g}")

    # ---------- 1000 Test Paths on [0,0.5] ----------
    R_TEST = 1000
    BASE_SEED_TEST = 999
    # Print the summary of Monte Carlo results
    mc_df = run_test_paths(
        R=R_TEST,
        base_seed=BASE_SEED_TEST,
        n_points_train=n_points_train,
        levels=levels,
        depth=depth,
        reg_strato=calib_res["reg_strato"],
        scaler_strato=calib_res["scaler_strato"],
        reg_ito=calib_res["reg_ito"],
        scaler_ito=calib_res["scaler_ito"],
        outdir=outdir
    )

    print_test_summary(mc_df)

    # ---------- Plot 3 random test paths ----------
    rng_plots = np.random.default_rng(42)
    n_random_plots = 3
    random_seeds = rng_plots.choice(R_TEST, size=n_random_plots, replace=False)

    for s in random_seeds:
        plot_test_path_for_seed(
            seed_idx=int(s),
            base_seed=BASE_SEED_TEST,
            n_points_train=n_points_train,
            levels=levels,
            depth=depth,
            alpha=alpha,
            reg_strato=calib_res["reg_strato"],
            scaler_strato=calib_res["scaler_strato"],
            reg_ito=calib_res["reg_ito"],
            scaler_ito=calib_res["scaler_ito"],
            outdir=outdir,
            suffix="random"
        )
    # Final output summary
    print(f"\nSaved outputs in: {outdir}/")
    print(" - train_path_fit.png                     ")
    print(" - test_path_seed_*_random.png           )")
    print(" - mc_results_test_paths.csv              ")
    print(" - cantor_full.png, cantor_zoom_middle_third.png")
    
