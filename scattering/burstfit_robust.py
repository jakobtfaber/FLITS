"""
burstfit_robust.py
==================

Diagnostic helpers to check how **robust** a scattering fit is to bad or
missing frequency channels.

Functions
---------
* :func:`subband_consistency` – refit *each* of *N* sub‑bands with the
  *same* model key as the global winner and compare τ₁ GHz.
* :func:`leave_one_out_influence` – approximate Δχ² if a channel were
  dropped, again for the chosen model.
* :func:`spectral_index_evolution` – spectral index γ describes how flux varies with frequency (S ∝ ν^γ). In your model, it's assumed constant, but physical effects like frequency-dependent scattering can cause it to evolve across the burst.model.
* :func:`plot_influence` – convenience bar plot.

These utilities are *model‑agnostic* – pass the `model_key` that won your
BIC scan (e.g. "M2" or "M3").
"""
from __future__ import annotations

import warnings
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from burstfit import FRBModel, FRBFitter, FRBParams, build_priors

__all__ = [
    "subband_consistency",
    "leave_one_out_influence",
    "plot_influence",
    "time_frequency_drift_check",
    "bootstrap_uncertainties"
]

# -----------------------------------------------------------------------------
# Sub‑band consistency test
# -----------------------------------------------------------------------------

def subband_consistency(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    init: FRBParams,
    *,
    model_key: str = "M3",
    n_sub: int = 4,
    n_steps: int = 600,
    pool = None,
) -> List[Tuple[float, float]]:
    """Return ``[(τ_mean, τ_std), …]`` for each frequency sub-band.

    The *same* `model_key` is used as in the global best-fit.  
    Pass the `init` that was used for the full fit so priors stay comparable.
    """
    if n_sub < 2:
        raise ValueError("Need at least two sub-bands")

    # If the chosen model has no scattering timescale, warn & return NaNs
    if model_key not in ("M2", "M3"):
        warnings.warn(
            f"subband_consistency: model_key='{model_key}' has no 'tau_1ghz'; "
            "returning NaNs for all sub-bands.",
            UserWarning
        )
        return [(np.nan, np.nan) for _ in range(n_sub)]

    # Compute sub-band edges in frequency index space
    edges = np.linspace(0, freq.size, n_sub + 1, dtype=int)
    out: List[Tuple[float, float]] = []

    for i in range(n_sub):
        sl = slice(edges[i], edges[i + 1])
        # Build a new model for this sub-band
        model = FRBModel(time=time, freq=freq[sl], data=data[sl], dm_init=dm_init)
        pri   = build_priors(init, scale=3.0)

        # Run an MCMC sample on this sub-band
        sampler = FRBFitter(model, pri, n_steps=n_steps, pool=pool).sample(init, model_key)

        # Flatten chain after discarding first quarter and thinning by 4
        chain = sampler.get_chain(discard=n_steps // 4, thin=4, flat=True)

        # Build parameter list to locate tau_1ghz
        param_names = ["c0", "t0", "gamma"]
        if model_key == "M3":
            param_names.append("zeta")
        param_names.append("tau_1ghz")

        tau_index = param_names.index("tau_1ghz")
        tau = chain[:, tau_index]

        out.append((float(np.mean(tau)), float(np.std(tau))))

    return out

# -----------------------------------------------------------------------------
# Leave‑one‑out influence diagnostic
# -----------------------------------------------------------------------------

def leave_one_out_influence(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    best_p: FRBParams,
    *,
    model_key: str = "M3",
    stride: int = 1,
) -> NDArray[np.floating]:
    """Δχ² influence per channel for the chosen `model_key`."""
    model = FRBModel(time=time, freq=freq, dm_init=dm_init)
    resid = data - model(best_p, model_key)
    chi2_full = np.sum(resid**2, axis=1)
    keep = slice(None, None, stride)
    return chi2_full[keep] - (chi2_full.sum() - chi2_full[keep])

# -----------------------------------------------------------------------------
# Quick bar‑plot helper
# -----------------------------------------------------------------------------

def plot_influence(ax, delta_chi2: NDArray[np.floating], freq: NDArray[np.floating]):
    import matplotlib.pyplot as plt

    sigma = np.std(delta_chi2)
    ax.bar(freq, delta_chi2, width=np.diff(freq).mean(), align="center", label="Δχ²")
    ax.axhline( 3 * sigma, ls="--", color="k", lw=1)
    ax.axhline(-3 * sigma, ls="--", color="k", lw=1)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Influence Δχ²")
    ax.set_title("Leave‑one‑out channel influence")
    ax.legend()
    plt.tight_layout()
    
# -----------------------------------------------------------------------------
# Check spectral index evolution
# -----------------------------------------------------------------------------

def spectral_index_evolution(
    data: NDArray[np.floating],
    freq: NDArray[np.floating], 
    time: NDArray[np.floating],
    window_ms: float = 1.0,
    min_snr: float = 3.0,
    stride_ms: float | None = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Track spectral index evolution across the burst duration.
    
    This function divides the burst into time windows and fits a power law
    (flux ∝ frequency^gamma) to each window's spectrum, allowing us to see
    if the spectral index changes over time.
    
    Parameters
    ----------
    data : array_like
        Dynamic spectrum (nfreq, ntime)
    freq : array_like
        Frequency axis in GHz
    time : array_like  
        Time axis in ms
    window_ms : float
        Time window size in milliseconds for each spectral fit
    min_snr : float
        Minimum SNR threshold for including time samples
    stride_ms : float, optional
        Step size between windows. If None, uses window_ms/2 (50% overlap)
        
    Returns
    -------
    times : array_like
        Center time of each window [ms]
    gammas : array_like
        Fitted spectral index for each window
    gamma_errs : array_like
        1-sigma uncertainty on each spectral index
        
    How it works
    ------------
    1. First identifies the "on-burst" region using SNR threshold
    2. Slides a window across the burst with specified stride
    3. For each window:
       - Averages the spectrum over the time window
       - Fits log(flux) = gamma * log(freq) + const
       - Records the spectral index gamma and its uncertainty
    4. This reveals if spectral index evolves (e.g., due to scattering)
    """
    dt_ms = time[1] - time[0]
    window_samples = int(window_ms / dt_ms)
    
    if stride_ms is None:
        stride_samples = window_samples // 2  # 50% overlap by default
    else:
        stride_samples = int(stride_ms / dt_ms)
    
    # Identify on-burst region using SNR
    time_profile = np.nansum(data, axis=0)
    time_profile /= np.nanmax(time_profile)
    noise_std = np.std(time_profile[:len(time_profile)//4])  # Use first quarter for noise
    burst_mask = time_profile > min_snr * noise_std
    
    if not np.any(burst_mask):
        return np.array([]), np.array([]), np.array([])
    
    # Find burst boundaries
    burst_indices = np.where(burst_mask)[0]
    burst_start = burst_indices[0]
    burst_end = burst_indices[-1]
    
    times = []
    gammas = []
    gamma_errs = []
    
    # Slide window across burst
    for start_idx in range(burst_start, burst_end - window_samples + 1, stride_samples):
        end_idx = start_idx + window_samples
        
        # Average spectrum in this window
        window_spectrum = np.mean(data[:, start_idx:end_idx], axis=1)
        
        # Only fit channels with significant signal
        spectrum_noise = np.std(data[:, :len(time)//4], axis=1).mean()
        good_channels = window_spectrum > min_snr * spectrum_noise
        
        if np.sum(good_channels) < 10:  # Need enough points for fit
            continue
        
        # Fit power law in log space: log(S) = gamma * log(f) + C
        log_freq = np.log10(freq[good_channels])
        log_flux = np.log10(window_spectrum[good_channels])
        
        # Use weighted least squares if we have channel uncertainties
        try:
            # Fit with numpy polyfit (returns [gamma, intercept])
            coeffs, cov = np.polyfit(log_freq, log_flux, 1, cov=True)
            gamma = coeffs[0]
            gamma_err = np.sqrt(cov[0, 0])
            
            times.append(time[start_idx + window_samples//2])  # Window center
            gammas.append(gamma)
            gamma_errs.append(gamma_err)
            
        except np.linalg.LinAlgError:
            # Singular matrix, skip this window
            continue
    
    return np.array(times), np.array(gammas), np.array(gamma_errs)

# -----------------------------------------------------------------------------
# Check DM optimization
# -----------------------------------------------------------------------------

def dm_optimization_check(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_center: float,
    dm_range: float = 10.0,
    n_trials: int = 21,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Check if the assumed DM is optimal by computing S/N vs DM.
    
    Parameters
    ----------
    data : array_like
        Dynamic spectrum (nfreq, ntime)
    freq : array_like
        Frequency axis [GHz]
    time : array_like
        Time axis [ms]
    dm_center : float
        Central DM value to search around [pc/cm^3]
    dm_range : float
        Range to search ±dm_range around center [pc/cm^3]
    n_trials : int
        Number of DM trials
        
    Returns
    -------
    dms : array_like
        Trial DM values
    snrs : array_like
        S/N for each trial DM
    """
    from burstfit import DM_DELAY_MS
    
    dms = np.linspace(dm_center - dm_range, dm_center + dm_range, n_trials)
    snrs = np.zeros(n_trials)
    
    dt = time[1] - time[0]
    
    for i, dm_trial in enumerate(dms):
        # Compute time delays relative to highest frequency
        delays = DM_DELAY_MS * (dm_trial - dm_center) * (freq**-2 - freq[-1]**-2)
        
        # Dedisperse by shifting each frequency channel
        data_dedispersed = np.zeros_like(data)
        for j, delay in enumerate(delays):
            shift = int(round(delay / dt))
            if abs(shift) < data.shape[1]:
                data_dedispersed[j] = np.roll(data[j], -shift)
        
        # Compute S/N as peak of summed profile over noise
        profile = np.sum(data_dedispersed, axis=0)
        noise = np.std(profile[:len(profile)//4])
        signal = np.max(profile)
        snrs[i] = signal / noise if noise > 0 else 0
    
    return dms, snrs

# -----------------------------------------------------------------------------
# Check the presence of scintillation
# -----------------------------------------------------------------------------

def scintillation_analysis(
    residual: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    n_freq_chunks: int = 8,
) -> Dict[str, Any]:
    """
    Analyze frequency structure in residuals to detect scintillation.
    
    Parameters
    ----------
    residual : array_like
        Residual dynamic spectrum (data - model)
    freq : array_like
        Frequency axis [GHz]
    time : array_like
        Time axis [ms]
    n_freq_chunks : int
        Number of frequency sub-bands for analysis
        
    Returns
    -------
    dict
        Contains:
        - acf_freq: Frequency autocorrelation function
        - decorr_bw: Decorrelation bandwidth [MHz]
        - modulation_index: Scintillation modulation index
        - freq_covariance: Frequency covariance matrix
    """
    # Get on-burst region
    residual = np.ma.masked_invalid(residual)
    burst_profile = np.nansum(np.abs(residual), axis=0)
    threshold = 0.2 * np.nanmax(burst_profile)
    burst_mask = burst_profile > threshold
    
    if np.sum(burst_mask) < 10:
        return {
            'acf_freq': np.array([]),
            'decorr_bw': np.nan,
            'modulation_index': np.nan,
            'freq_covariance': np.array([])
        }
    
    # Use on-burst residuals
    burst_residual = residual[:, burst_mask]
    
    # Compute frequency autocorrelation
    residual_spectrum = np.nanmean(burst_residual, axis=1)
    residual_spectrum -= np.nanmean(residual_spectrum)
    
    acf_freq = np.correlate(residual_spectrum, residual_spectrum, mode='same')
    acf_freq = acf_freq / acf_freq[len(acf_freq)//2]
    
    # Find decorrelation bandwidth (where ACF drops to 1/e)
    center = len(acf_freq) // 2
    half_acf = acf_freq[center:]
    decorr_idx = np.where(half_acf < 1/np.e)[0]
    
    if len(decorr_idx) > 0:
        df = freq[1] - freq[0]  # GHz
        decorr_bw = decorr_idx[0] * df * 1000  # Convert to MHz
    else:
        decorr_bw = np.nan
    
    # Compute modulation index
    mean_flux = np.mean(np.abs(burst_residual))
    std_flux = np.std(burst_residual)
    modulation_index = std_flux / mean_flux if mean_flux > 0 else np.nan
    
    # Frequency covariance matrix
    freq_chunks = np.array_split(burst_residual, n_freq_chunks, axis=0)
    chunk_means = [np.mean(chunk, axis=1) for chunk in freq_chunks]
    freq_covariance = np.cov(chunk_means)
    
    return {
        'acf_freq': acf_freq,
        'decorr_bw': decorr_bw,
        'modulation_index': modulation_index,
        'freq_covariance': freq_covariance
    }

# -----------------------------------------------------------------------------
# Check for time-frequency drift
# -----------------------------------------------------------------------------

def time_frequency_drift_check(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    model: NDArray[np.floating],
) -> Dict[str, Any]:
    """
    Check for unmodeled time-frequency drift in residuals.
    
    Parameters
    ----------
    data, model : array_like
        Observed and model dynamic spectra
    freq, time : array_like
        Frequency [GHz] and time [ms] axes
        
    Returns
    -------
    dict
        Contains:
        - drift_rate: Fitted drift rate [MHz/ms]
        - drift_snr: Significance of drift detection
        - drift_corrected_residual: Residual after drift removal
    """
    residual = data - model
    
    # Find peak time at each frequency
    peak_times = np.zeros(len(freq))
    peak_snrs = np.zeros(len(freq))
    
    for i in range(len(freq)):
        profile = residual[i]
        if np.std(profile) > 0:
            # Use cross-correlation with model to find delay
            model_profile = model[i]
            xcorr = np.correlate(profile, model_profile, mode='same')
            peak_idx = np.argmax(np.abs(xcorr))
            peak_times[i] = time[peak_idx]
            peak_snrs[i] = np.max(np.abs(profile)) / np.std(profile[:len(profile)//4])
    
    # Fit linear drift to high-SNR channels
    mask = peak_snrs > 3
    if np.sum(mask) > 10:
        # Fit: time = drift_rate * freq + offset
        p = np.polyfit(freq[mask], peak_times[mask], 1, w=peak_snrs[mask])
        drift_rate = p[0] * 1000  # Convert GHz/ms to MHz/ms
        
        # Compute significance
        residuals_fit = peak_times[mask] - np.polyval(p, freq[mask])
        chi2 = np.sum((residuals_fit / (1/peak_snrs[mask]))**2)
        drift_snr = np.abs(drift_rate) / (np.std(residuals_fit) * 1000)
        
        # Correct for drift
        drift_corrected = np.zeros_like(residual)
        dt = time[1] - time[0]
        for i in range(len(freq)):
            shift = int(round((p[0] * freq[i] + p[1] - time[len(time)//2]) / dt))
            if abs(shift) < len(time):
                drift_corrected[i] = np.roll(residual[i], -shift)
    else:
        drift_rate = 0.0
        drift_snr = 0.0
        drift_corrected = residual
    
    return {
        'drift_rate': drift_rate,
        'drift_snr': drift_snr,
        'drift_corrected_residual': drift_corrected
    }

# -----------------------------------------------------------------------------
# Perform bootstrap error analysis
# -----------------------------------------------------------------------------

def bootstrap_uncertainties(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    best_params: FRBParams,
    model_key: str,
    n_bootstrap: int = 100,
    n_steps: int = 500,
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate parameter uncertainties using bootstrap resampling.
    
    Parameters
    ----------
    data : array_like
        Original dynamic spectrum
    freq, time : array_like
        Frequency and time axes
    dm_init : float
        Initial DM
    best_params : FRBParams
        Best-fit parameters from original fit
    model_key : str
        Model to use
    n_bootstrap : int
        Number of bootstrap samples
    n_steps : int
        MCMC steps per bootstrap (keep short)
        
    Returns
    -------
    dict
        Parameter names -> (mean, std) from bootstrap
    """
    from burstfit import FRBModel, FRBFitter, build_priors
    
    # Get residuals for resampling
    model = FRBModel(time=time, freq=freq, dm_init=dm_init)
    best_model = model(best_params, model_key)
    residuals = data - best_model
    
    # Store bootstrap results
    param_names = list(best_params.__dict__.keys())
    bootstrap_results = {name: [] for name in param_names}
    
    print(f"Running {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        # Resample residuals with replacement
        n_freq, n_time = residuals.shape
        boot_indices = np.random.randint(0, n_freq, size=n_freq)
        boot_residuals = residuals[boot_indices]
        
        # Create bootstrap dataset
        boot_data = best_model + boot_residuals
        
        # Fit bootstrap sample
        boot_model = FRBModel(time=time, freq=freq, data=boot_data, dm_init=dm_init)
        priors = build_priors(best_params, scale=2.0)  # Tighter priors
        
        fitter = FRBFitter(boot_model, priors, n_steps=n_steps)
        sampler = fitter.sample(best_params, model_key)
        
        # Get median parameters from short chain
        chain = sampler.get_chain(discard=n_steps//4, flat=True)
        medians = np.median(chain, axis=0)
        
        # Store results
        for j, name in enumerate(param_names[:len(medians)]):
            bootstrap_results[name].append(medians[j])
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_bootstrap}")
    
    # Compute statistics
    uncertainties = {}
    for name in param_names:
        if name in bootstrap_results and len(bootstrap_results[name]) > 0:
            values = np.array(bootstrap_results[name])
            uncertainties[name] = (np.mean(values), np.std(values))
        else:
            uncertainties[name] = (getattr(best_params, name), 0.0)
    
    return uncertainties