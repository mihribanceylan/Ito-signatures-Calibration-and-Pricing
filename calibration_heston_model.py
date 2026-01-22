
import os, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import iisignature
import esig


# ---------------------------
# Heston model simulation
# ---------------------------
def correlated_bms_correct(N, rho, rng, t_final=1.0, t_0=0.0):
    """
    Simulates two correlated Brownian motions B and W with correlation `rho`.

    Parameters:
    - N: Number of time steps
    - rho: Correlation between the Brownian motions B and W
    - rng: Random number generator
    - t_final: Final time for the simulation
    - t_0: Initial time

    Returns:
    - t: Time grid
    - B: First Brownian motion (correlated)
    - W: Second Brownian motion (correlated)
    """
    t = np.linspace(t_0, t_final, num=N)
    dt = abs(t[1] - t[0])
    dB = rng.normal(0.0, np.sqrt(dt), N); dB[0] = 0.0
    dW = rho * dB + rng.normal(0.0, np.sqrt(dt), N) * np.sqrt(max(1.0 - rho**2, 0.0)); dW[0] = 0.0
    return t, np.cumsum(dB), np.cumsum(dW)

def Simulate_Heston(N, S0, mu, a, kappa, theta, V0, rho, rng, t_final=1.0, t_0=0.0):
    """
    Simulates the Heston model for asset prices (S) and volatility (V).

    Parameters:
    - N: Number of time steps
    - S0: Initial asset price
    - mu: Drift of the asset price
    - a, kappa, theta, V0, rho: Parameters of the Heston model 
    - rng: Random number generator

    Returns:
    - S: Simulated asset prices
    - V: Simulated volatility
    - B: Brownian motion B
    - W: Brownian motion W
    - t: Time grid
    """

    S = np.zeros(N); V = np.zeros(N)
    S[0], V[0] = S0, V0
    t, B, W = correlated_bms_correct(N, rho, rng, t_final, t_0)
    dt = (t_final - t_0) / (N - 1)
    for k in range(N-1):
        dB = B[k+1]-B[k]; dW = W[k+1]-W[k]
        V[k+1] = V[k] + kappa*(theta - V[k])*dt + a*np.sqrt(V[k])*dW
        S[k+1] = S[k] + S[k]*mu*dt         + np.sqrt(V[k])*S[k]*dB
    return S, V, B, W, t

def reconstruct_Q_BMs_from_SV(S, V, a):
    """
    Reconstructs Brownian motions B^Q and W^Q from the discretized Heston paths (S, V).
    
    Parameters:
    - S: Asset prices from Heston simulation
    - V: Volatility from Heston simulation
    - a: Volatility parameter

    Returns:
    - B_ext: Reconstructed Brownian motion B^Q
    - W_ext: Reconstructed Brownian motion W^Q
    """
    N = len(S)
    B_ext = np.zeros(N)
    W_ext = np.zeros(N)

    for k in range(N-1):

        dS = S[k+1] - S[k]
        dV = V[k+1] - V[k]

        diff_S = np.sqrt(V[k]) * S[k]
        diff_V = a * np.sqrt(V[k])

        dBq = dS / diff_S 
        dWq = dV / diff_V 
        
        B_ext[k+1] = B_ext[k] + dBq
        W_ext[k+1] = W_ext[k] + dWq

    return B_ext, W_ext

# ---------------------------
# Streams (t,B,W)
# ---------------------------
def augment_noises(B, W):
    """
    Augments the noises for the Stratonovich and Itô signatures, adding time as the first column.

    Parameters:
    - B: Brownian motion B
    - W: Brownian motion W

    Returns:
    - Augmented data with time, B, and W
    """
    n = len(W)
    t_scaled = np.arange(n, dtype=float)/max(n, 1)  
    return np.column_stack([t_scaled, B, W])        


# ---------------------------
# Helper functions for Keys / Words 
# ---------------------------
def parse_key_to_word(key_str):
    """
    Converts a string key (e.g., '(2,3)') to a tuple (e.g., (2, 3)).
    
    Parameters:
    - key_str: String representation of a key
    
    Returns:
    - The corresponding tuple
    """
    if key_str == "()":
        return ()
    obj = ast.literal_eval(key_str)
    if isinstance(obj, int):
        return (obj,)
    return tuple(obj)

def format_word_to_key(word):

    """
    Converts a tuple (e.g., (2, 3)) into a string key (e.g., '(2,3)').
    
    Parameters:
    - word: Tuple representing a key
    
    Returns:
    - The corresponding string representation of the key
    """
    if len(word) == 0:
        return "()"
    if len(word) == 1:
        return f"({word[0]})"
    return "(" + ",".join(str(x) for x in word) + ")"


# ---------------------------
# Stratonovich Signature 
# ---------------------------
def build_prefix_sig_df_strato(noises, depth_sig):
    """
    Builds the Stratonovich signature of (t, B, W) up to the specified depth, returning a DataFrame.
    
    Parameters:
    - noises: Augmented noise data (time, B, W)
    - depth_sig: Depth of the signature
    
    Returns:
    - DataFrame containing the Stratonovich signature
    """
    T, d = noises.shape
    keys_raw = esig.sigkeys(d, depth_sig).strip().split()
    keys = keys_raw if (keys_raw and keys_raw[0] == "()") else ["()"] + keys_raw
    rows = []
    for i in range(1, T+1):
        v = iisignature.sig(noises[:i, :], depth_sig)
        v = np.concatenate(([1.0], v))  
        rows.append(v)
    return pd.DataFrame(rows, columns=keys), keys


# ---------------------------
# Tilde-Basis (Strato -> Itô transformation for B-component)
# ---------------------------
def tilde_transformation(word):
    w = list(word)
    last = w[-1] #last component of the word

    if last == 1: #if last component is time
        w1 = w + [2]   # attach B
        w2 = w + [3]   # attach W
        return [w1, w2]
    else:
        w_aux  = w + [2] # attach B
        w_aux2 = w + [3] # attach W
        w3 = w.copy()    #copy word w
        w3[-1] = 1     # and exchange last index by time index
        return [w_aux, w_aux2, w3]
    
def build_tilde_basis_from_strato_df(sig_df, order_model, rho):
    """
    Builds the tilde basis from a Stratonovich signature DataFrame.
    
    Parameters:
    - sig_df: Signature DataFrame for Stratonovich signatures
    - order_model: Order of the model (depth of the signature)
    - rho: Correlation between assets
    
    Returns:
    - DataFrame of the tilde-transformed basis
    """
    d = 3 

    keys_n = esig.sigkeys(d, order_model).strip().split()
    words_n = [] # list of indices
    for k in keys_n:
        val = ast.literal_eval(k)# array
        if isinstance(val, int): #if val integer, save [val]
            words_n.append([val])
        else:
            words_n.append(list(val)) # else create a list,i.e., val=(1,2)--> [1,2]

    tilde_list = [] #create list with tilde transformation
    for k in range(1, len(words_n)):
        tilde_list.append(tilde_transformation(words_n[k]))

    aus_B = [] #create e_tilde
    for k, w in enumerate(words_n):
        if k == 0: #empty set equal to B
            feat = sig_df["(2)"]   # B_t
        else:
            tilde = tilde_list[k-1]
            last = w[-1]

            if last == 1: #last index time
                key0 = format_word_to_key(tilde[0]) 
                feat = sig_df[key0]

            elif last == 2: #last index is B
                key0 = format_word_to_key(tilde[0]) #ends with B
                key2 = format_word_to_key(tilde[2]) #w' 0 #tilde[2] corresponds to the time-component in tilde_transfo
                feat = sig_df[key0] - 0.5 * sig_df[key2] #The correction which we need for strato model

            elif last == 3:#last index W
                key0 = format_word_to_key(tilde[0])# ends with B
                key2 = format_word_to_key(tilde[2])# ends with 0
                feat = sig_df[key0] - 0.5 * rho * sig_df[key2] #correction for strato

            else:
                continue

        aus_B.append(feat)

    new_keys_B = [keys_n[k] + "~B" for k in range(len(words_n))] #column names
    tilde_df = pd.DataFrame({name: series.values for name, series in zip(new_keys_B, aus_B)}) #Dataframe with columnnames
    return tilde_df, new_keys_B


# ---------------------------
# Itô from Stratonovich via Time Channel + Filter for "ends with B"
# ---------------------------
def _non_adjacent_subsets(indices):
    """
    Generates non-adjacent subsets of the given indices.
    
    Parameters:
    - indices: List of indices
    
    Returns:
    - List of non-adjacent subsets
    """
    out=[[]]
    for u in indices:
        out += [sub+[u] for sub in out if (not sub) or (u - sub[-1] > 1)]# new index is at least 2 larger, non-adjacent
    return out[1:]

def generate_ito_time_correction_indices(keys, rho):
    """
    Generates time correction indices for the Itô signature from Stratonovich signatures.
    
    Parameters:
    - keys: List of Stratonovich signature keys
    - rho: Correlation between assets
    
    Returns:
    - corr: A dictionary of time correction indices and corresponding coefficients
    """
    k2i = {k:i for i,k in enumerate(keys)}
    t_ch, B_ch, W_ch = 1, 2, 3
    corr = {}
    for i, key in enumerate(keys):
        if len(key) < 2:
            corr[i] = []; continue 

        idxs = tuple(int(x) for x in key)
        cand = [j for j in range(len(idxs)-1)
                if idxs[j] in (B_ch, W_ch) and idxs[j+1] in (B_ch, W_ch)] 
        combos=[]
        for subset in _non_adjacent_subsets(cand):
            new_idx=[]; skip=False
            coef = 1.0
            for j in range(len(idxs)):
                if skip:
                    skip=False
                    continue
                if j in subset:
                    a, b = idxs[j], idxs[j+1]
                    coef *= -0.5
                    if {a,b} == {B_ch, W_ch} and a != b:
                        coef *= rho
                    new_idx.append(t_ch)
                    skip=True
                else:
                    new_idx.append(idxs[j])

            new_key = tuple(new_idx)
            jpos = k2i.get(new_key)
            if jpos is not None:
                combos.append((jpos, coef))
        corr[i] = combos
    return corr

def build_prefix_sig_df_ito_time(noises, depth_sig, rho):
    """
    Builds the Itô signature using time correction for Stratonovich signatures.
    
    Parameters:
    - noises: Augmented noise data (time, B, W)
    - depth_sig: Depth of the signature
    - rho: Correlation between assets
    
    Returns:
    - DataFrame containing the Itô signature
    """
    T, d = noises.shape
    keys_raw = esig.sigkeys(d, depth_sig).strip().split()
    keys = keys_raw if (keys_raw and keys_raw[0] == "()") else ["()"] + keys_raw
    keys_tuples = [parse_key_to_word(k) for k in keys]  
    corr_map = generate_ito_time_correction_indices(keys_tuples, rho)

    rows=[]
    for i in range(1, T+1):
        s = noises[:i, :]
        strat_no1 = iisignature.sig(s, depth_sig)
        strat = np.concatenate(([1.0], strat_no1))
        ito = strat.copy()
        for idx in range(len(keys)):
            for jpos, coef in corr_map[idx]:
                ito[idx] += coef * strat[jpos]
        rows.append(ito)
    return pd.DataFrame(rows, columns=keys), keys

def select_ito_features_ending_in_B(ito_df):
    """
    Selects Itô features that end with B.
    
    Parameters:
    - ito_df: DataFrame containing the Itô signature
    
    Returns:
    - A subset of the DataFrame containing only features that end with B
    """
    selected_cols = []
    for col in ito_df.columns:
        if col == "()":
            continue
        word = parse_key_to_word(col)
        if len(word) > 0 and word[-1] == 2:  # last index B
            selected_cols.append(col)
    return ito_df[selected_cols], selected_cols


# ---------------------------
# Lasso: Calibration on Full Path & Application to New Paths
# ---------------------------
def fit_lasso_full_path(X_df, y, alpha=1e-5):
    """
    Fits a Lasso regression model to the full training path.
    
    Parameters:
    - X_df: DataFrame of features
    - y: Target values (e.g., Heston price paths)
    - alpha: Regularization strength
    
    Returns:
    - Dictionary containing the fitted model, scaler, predicted values, and training MSE
    """
    X = X_df.values #numpy array
    scaler = StandardScaler(with_mean=True, with_std=True) 
    X_s = scaler.fit_transform(X)

    reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    reg.fit(X_s, y) #saves parameter and intercept

    yhat = reg.predict(X_s) #prediction with regressed parameters
    mse_train = mean_squared_error(y, yhat)

    return {
        "reg": reg,
        "scaler": scaler,
        "yhat_train": yhat,
        "mse_train": mse_train
    }

def apply_lasso_model(X_df, y, reg, scaler):
    """
    Applies a trained Lasso model to new data and computes the MSE.
    
    Parameters:
    - X_df: DataFrame of new feature data
    - y: True values for the new data
    - reg: The trained Lasso model
    - scaler: The scaler used during training
    
    Returns:
    - MSE and predicted values for the new data
    """
    # with trained model approximate new path and calculate test-MSE
    
    X = X_df.values #values of test-path, the signatures
    X_s = scaler.transform(X)
    yhat = reg.predict(X_s)
    mse = mean_squared_error(y, yhat)
    return mse, yhat


# ---------------------------
#  Calibration on a Single Training Path 
# ---------------------------
def calibrate_on_single_path(rng, N_train, params, order_model, alpha, outdir):
    """
    Performs calibration on a single training path (T_train = 1.0) using Lasso.
    
    Parameters:
    - rng: Random number generator for reproducibility
    - N_train: Number of training points (time steps)
    - params: Parameters for the Heston model (S0, V0, mu, a, kappa, theta, rho)
    - order_model: Order of the model (depth of the signature)
    - alpha: Regularization parameter for Lasso
    - outdir: Directory to save output plots
    
    Returns:
    - Dictionary containing calibration results
    """
    S0, V0, mu, a, kappa, theta, rho = params
    # Simulate training path
    S_tr, V_tr, B_tr, W_tr, t_tr = Simulate_Heston(
        N_train, S0, mu, a, kappa, theta, V0, rho, rng=rng, t_final=1.0
    )

    # Reconstruct Q-BMs (B_ext, W_ext) from (S_tr, V_tr)
    B_ext_tr, W_ext_tr = reconstruct_Q_BMs_from_SV(S_tr, V_tr, a)

    # Augment the data with time and B, W components
    noises_tr = augment_noises(B_ext_tr, W_ext_tr)
    # Build Stratonovich signature
    depth_sig = order_model + 1
    StratSig_df_tr, _ = build_prefix_sig_df_strato(noises_tr, depth_sig=depth_sig)
    Strat_tilde_df_tr, tilde_keys_tr = build_tilde_basis_from_strato_df(StratSig_df_tr, order_model, rho)
    # Build Itô signature (with time correction)
    ItoSig_df_tr, _ = build_prefix_sig_df_ito_time(noises_tr, depth_sig=depth_sig, rho=rho)
    Ito_B_df_tr, ito_B_keys_tr = select_ito_features_ending_in_B(ItoSig_df_tr)
    # Fit Lasso models for Stratonovich and Itô
    res_strat = fit_lasso_full_path(Strat_tilde_df_tr, S_tr, alpha=alpha)
    res_ito   = fit_lasso_full_path(Ito_B_df_tr,      S_tr, alpha=alpha)
    # Plot the training path and the fits
    plt.figure(figsize=(11,6))
    plt.plot(t_tr, S_tr,                    label="Heston (Train)", color="black", lw=1.6)
    plt.plot(t_tr, res_strat["yhat_train"], label="Stratonovich", alpha=0.9)
    plt.plot(t_tr, res_ito["yhat_train"],   label="Itô", alpha=0.9, linestyle='dashed')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "train_path_fit.png"), dpi=150)
    plt.close()

    return {
        "t_train": t_tr,
        "S_train": S_tr,
        "mse_strato_train": res_strat["mse_train"],
        "mse_ito_train":    res_ito["mse_train"],
        "reg_strato":       res_strat["reg"],
        "scaler_strato":    res_strat["scaler"],
        "reg_ito":          res_ito["reg"],
        "scaler_ito":       res_ito["scaler"],
        "tilde_keys":       tilde_keys_tr,
        "ito_B_keys":       ito_B_keys_tr
    }


# ---------------------------
# Monte-Carlo on test paths (T_test = 0.5)
# ---------------------------
def summarize_mc(df, col):
    """
    Summarizes Monte Carlo results for a given column.
    Computes the mean, standard deviation, standard error, and 95% confidence interval.
    
    Parameters:
    - df: DataFrame containing the results.
    - col: Column name for which the statistics are computed.
    
    Returns:
    - m: Mean of the column.
    - sd: Standard deviation.
    - ci: 95% confidence interval.
    """
    n  = len(df)
    m  = df[col].mean()
    sd = df[col].std(ddof=1)
    se = sd / np.sqrt(max(n,1))
    ci = (m - 1.96*se, m + 1.96*se)
    return m, sd, ci

def run_test_paths(R, base_seed, N_train, params, order_model,
                   reg_strato, scaler_strato, reg_ito, scaler_ito,
                   outdir):
    """
    Run Monte Carlo simulations for test paths.
    It simulates R test paths, applies Lasso regression to Stratonovich and Itô signatures,
    and computes the MSE for both.
    
    Parameters:
    - R: Number of test paths to simulate.
    - base_seed: Base seed for random number generation.
    - N_train: Number of training points.
    - params: Model parameters (S0, V0, mu, a, kappa, theta, rho).
    - order_model: Order of the model (depth of the signature).
    - reg_strato, scaler_strato, reg_ito, scaler_ito: Trained models and scalers for Stratonovich and Itô.
    - outdir: Output directory for saving the results.
    
    Returns:
    - mc_df: DataFrame containing MSE results for each test path.
    """
    S0, V0, mu, a, kappa, theta, rho = params

    rows = []
    depth_sig = order_model + 1

    # Set the test horizon (T_test = 0.5)
    N_test = int(0.5 * (N_train - 1)) + 1
    T_test = 0.5

    for r in range(R):
        rng_r = np.random.default_rng(base_seed + r)

        # Simulate test path on [0, 0.5] with N_test points
        S_te, V_te, B_te, W_te, t_te = Simulate_Heston(
            N_test, S0, mu, a, kappa, theta, V0, rho, rng=rng_r, t_final=T_test
        )

        # Reconstruct the Q-Brownian motions (B_ext, W_ext) from the simulated S and V
        B_ext_te, W_ext_te = reconstruct_Q_BMs_from_SV(S_te, V_te, a)
        # Augment the noise process (t, B, W)
        noises_te = augment_noises(B_ext_te, W_ext_te)
        # Build Stratonovich and Itô signatures for the test path
        StratSig_df_te, _ = build_prefix_sig_df_strato(noises_te, depth_sig=depth_sig)
        Strat_tilde_df_te, _ = build_tilde_basis_from_strato_df(StratSig_df_te, order_model, rho)

        ItoSig_df_te, _ = build_prefix_sig_df_ito_time(noises_te, depth_sig=depth_sig, rho=rho)
        Ito_B_df_te, _ = select_ito_features_ending_in_B(ItoSig_df_te)
        # Apply Lasso models to both Stratonovich and Itô signatures
        mseS_te, yhatS_te = apply_lasso_model(Strat_tilde_df_te, S_te, reg_strato, scaler_strato)
        mseI_te, yhatI_te = apply_lasso_model(Ito_B_df_te,      S_te, reg_ito,    scaler_ito)
        # Calculate the difference in MSEs for Stratonovich and Itô
        diff = mseS_te - mseI_te  # >0: Itô besser
        rows.append({
            "seed_index": r,
            "mse_strato_test": mseS_te,
            "mse_ito_test":    mseI_te,
            "diff_test":       diff
        })
    # Create a DataFrame to store the MSE results from all test paths
    mc_df = pd.DataFrame(rows)
    mc_df.to_csv(os.path.join(outdir, "mc_results_test_paths.csv"), index=False)

    return mc_df

def print_summary(mc_df):
    """
    Print the Monte Carlo summary including mean, standard deviation, and confidence intervals 
    for the MSE of Stratonovich, Itô, and the difference.
    
    Parameters:
    - mc_df: DataFrame containing the MSE results from the Monte Carlo simulations.
    """
    mS, sS, ciS = summarize_mc(mc_df, "mse_strato_test")
    mI, sI, ciI = summarize_mc(mc_df, "mse_ito_test")
    md, sd, cid = summarize_mc(mc_df, "diff_test")
    print(f"\n=== Monte-Carlo summary (OUT-OF-SAMPLE / TEST MSE, T=0.5) ===")
    print(f"Strato-tilde  mean={mS:.3e}  sd={sS:.3e}  95%CI=[{ciS[0]:.3e}, {ciS[1]:.3e}]")
    print(f"Itô (B-end)   mean={mI:.3e}  sd={sI:.3e}  95%CI=[{ciI[0]:.3e}, {ciI[1]:.3e}]")
    print(f"Diff (Strato-Itô) mean={md:.3e} sd={sd:.3e}  95%CI=[{cid[0]:.3e}, {cid[1]:.3e}]")


def plot_test_path_for_seed(seed_idx, base_seed, N_train, params, order_model,
                            reg_strato, scaler_strato, reg_ito, scaler_ito,
                            outdir, suffix="random"):
    """
    Plot a test path for a given seed index (T_test = 0.5).
    
    Parameters:
    - seed_idx: The seed index for the test path.
    - base_seed: Base seed for random number generation.
    - N_train: Number of training points.
    - params: Model parameters (S0, V0, mu, a, kappa, theta, rho).
    - order_model: Order of the model (depth of the signature).
    - reg_strato, scaler_strato, reg_ito, scaler_ito: Trained models and scalers.
    - outdir: Output directory to save the plot.
    - suffix: Suffix for the plot filename (default is "random").
    """
    S0, V0, mu, a, kappa, theta, rho = params
    depth_sig = order_model + 1

    # Set the test horizon (T_test = 0.5)
    N_test = int(0.5 * (N_train - 1)) + 1
    T_test = 0.5

    rng_r = np.random.default_rng(base_seed + seed_idx)
    S_te, V_te, B_te, W_te, t_te = Simulate_Heston(
        N_test, S0, mu, a, kappa, theta, V0, rho, rng=rng_r, t_final=T_test
    )
    # Reconstruct the Q-BMs
    B_ext_te, W_ext_te = reconstruct_Q_BMs_from_SV(S_te, V_te, a)
    noises_te = augment_noises(B_ext_te, W_ext_te)
     # Build the signatures for Stratonovich and Itô
    StratSig_df_te, _ = build_prefix_sig_df_strato(noises_te, depth_sig=depth_sig)
    Strat_tilde_df_te, _ = build_tilde_basis_from_strato_df(StratSig_df_te, order_model, rho)

    ItoSig_df_te, _ = build_prefix_sig_df_ito_time(noises_te, depth_sig=depth_sig, rho=rho)
    Ito_B_df_te, _ = select_ito_features_ending_in_B(ItoSig_df_te)
    # Apply Lasso models to the Stratonovich and Itô signatures
    mseS_te, yhatS_te = apply_lasso_model(Strat_tilde_df_te, S_te, reg_strato, scaler_strato)
    mseI_te, yhatI_te = apply_lasso_model(Ito_B_df_te,      S_te, reg_ito,    scaler_ito)
    # Plot the results
    plt.figure(figsize=(11,6))
    plt.plot(t_te, S_te,       label="Heston (Test, T=0.5)", color="black", lw=1.6)
    plt.plot(t_te, yhatS_te,   label="Stratonovich", alpha=0.9)
    plt.plot(t_te, yhatI_te,   label="Itô ", alpha=0.9, linestyle='dashed')
    plt.legend(loc='upper left')
    plt.tight_layout()
    fname = os.path.join(outdir, f"test_path_seed_{seed_idx}_{suffix}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# =========================
# Main: Run the Calibration and Monte-Carlo Simulation
# =========================
if __name__ == "__main__":
    outdir = "calib_outputs_heston"
    os.makedirs(outdir, exist_ok=True)

    #  --- Set up parameters ---
    ALPHA        = 1e-5
    ORDER_MODEL  = 2      
    N_TRAIN      = 2000  
    params = (1.0, 0.08, 0.0001, 0.25, 0.5, 0.15, -0.5)  # (S0,V0,mu,a,kappa,theta,rho)

    # ---------- Calibration on a single training path (T=1) ----------
    CALIB_SEED = 123
    rng_calib = np.random.default_rng(CALIB_SEED)
    calib_res = calibrate_on_single_path(
        rng=rng_calib, N_train=N_TRAIN, params=params,
        order_model=ORDER_MODEL, alpha=ALPHA,
        outdir=outdir
    )
    # Print results of the calibration on a single path
    print("\n=== Calibration ) ===")
    print(f"Stratonovich-tilde  MSE_train={calib_res['mse_strato_train']:.6g}")
    print(f"Itô (B-end)         MSE_train={calib_res['mse_ito_train']:.6g}")

    # ---------- 1000 Test Paths on [0,0.5] ----------
    R_TEST = 1000
    BASE_SEED_TEST = 777
    # Print the summary of Monte Carlo results
    mc_df = run_test_paths(
        R=R_TEST, base_seed=BASE_SEED_TEST,
        N_train=N_TRAIN, params=params,
        order_model=ORDER_MODEL, 
        reg_strato=calib_res["reg_strato"], scaler_strato=calib_res["scaler_strato"],
        reg_ito=calib_res["reg_ito"],       scaler_ito=calib_res["scaler_ito"],
        outdir=outdir
    )

    print_summary(mc_df)

    # ---------- Plot 3 random test paths ----------
    rng_plots = np.random.default_rng(42)
    n_random_plots = 3
    random_seeds = rng_plots.choice(R_TEST, size=n_random_plots, replace=False)
    
    for s in random_seeds:
        plot_test_path_for_seed(
            seed_idx=int(s),
            base_seed=BASE_SEED_TEST,
            N_train=N_TRAIN, params=params,
            order_model=ORDER_MODEL,
            reg_strato=calib_res["reg_strato"], scaler_strato=calib_res["scaler_strato"],
            reg_ito=calib_res["reg_ito"],       scaler_ito=calib_res["scaler_ito"],
            outdir=outdir,
            suffix="random"
        )
    # Final output summary
    print(f"\nSaved outputs in: {outdir}/")
    print(" - train_path_fit.png        ")
    print(" - test_path_seed_*_random.png ")
    print(" - mc_results_test_paths.csv   ")
    