import numpy as np

# ==============================
# Payoffs 
# ==============================
def log_increments(S_time_log):
    """
    Calculates the log price increments.

    Parameters:
    - S_time_log: Time-stamped log price data

    Returns:
    - Log increments of the prices
    """
    return S_time_log[:,1:,1:] - S_time_log[:,:-1,1:]

def realized_variance_from_logs(S_time_log):
    """
    Calculates the realized variance from the log price increments.
    """
    dX = log_increments(S_time_log)
    return np.sum(dX**2,axis=1)

def realized_vol_from_logs(S_time_log):
    """
    Calculates the realized volatility from the log price increments.
    """
    return np.sqrt(realized_variance_from_logs(S_time_log))

def correlation_12(S_time_log):
    """
    Calculates the correlation between the two assets' log price increments.
    """
    dX = log_increments(S_time_log)
    x = dX[:,:,0]; y = dX[:,:,1]
    num = np.sum(x*y, axis=1)
    den = np.sqrt(np.sum(x*x, axis=1) * np.sum(y*y, axis=1))
    den = np.where(den>1e-16, den, np.inf)
    return num/den

def covariance_basket(S_time_log):
    """
    Calculates the covariance between the two assets' log price increments.

    """
    dX = log_increments(S_time_log)
    return np.sum(dX[:,:,0]*dX[:,:,1], axis=1)

def rv_call_from_logs(S_time_log, K):
    """
    Calculates the realized variance call payoff from the log price increments.
    """
    RV = realized_vol_from_logs(S_time_log)
    return np.maximum(RV - K, 0.0)

def rv_swap_from_logs(S_time_log, K):
    """
    Calculates the realized variance swap payoff from the log price increments.

    """
    RVar = realized_variance_from_logs(S_time_log)
    return RVar - K

def corr_swap_payoff(S_time_log, K):
    """
    Calculates the correlation swap payoff from the log price increments.
    """
    return correlation_12(S_time_log) - K

def corr_call_payoff(S_time_log, K):
    """
    Calculates the correlation call payoff from the log price increments.
    """
    return np.maximum(correlation_12(S_time_log) - K, 0.0)

def cov_swap_payoff(S_time_log, K):
    """
    Calculates the covariance swap payoff from the log price increments.
    """
    return covariance_basket(S_time_log) - K

def cov_call_payoff(S_time_log, K):
    """
    Calculates the covariance call payoff from the log price increments.
    """
    return np.maximum(covariance_basket(S_time_log) - K, 0.0)
