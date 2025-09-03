import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants as const

# Assume these are defined elsewhere in your script
# from baseband_analysis.core.bbdata import BBData
# from baseband_analysis.core.dedispersion import delay_across_the_band

# Dispersion constant in MHz^2 pc^-1 cm^3 s
K_DM = 4.148808e3

# Observatory coordinates as module-level constants
CHIME_LOCATION = EarthLocation(
    lat=49.3206 * u.deg, lon=-119.6236 * u.deg, height=545 * u.m
)
DSA_LOCATION = EarthLocation(
    lat=37.2333 * u.deg, lon=-118.2834 * u.deg, height=1222 * u.m
)

def calculate_dm_timing_error(dm_uncertainty, f_obs, f_ref):
    """
    Calculates the timing error due to DM uncertainty.

    Parameters
    ----------
    dm_uncertainty : float
        The uncertainty in the Dispersion Measure (pc/cm^3).
    f_obs : astropy.units.Quantity
        The central observing frequency in MHz.
    f_ref : astropy.units.Quantity
        The reference frequency in MHz.

    Returns
    -------
    astropy.units.Quantity
        The timing error in milliseconds.
    """
    # Calculate the time shift in seconds
    time_shift = K_DM * dm_uncertainty * (1 / f_obs.value**2 - 1 / f_ref.value**2) * u.s
    
    # Return the absolute value in milliseconds
    return np.abs(time_shift.to(u.ms))


def compute_toa(t0, offset, f_center, DM, f_ref):
    """Compute a time of arrival referenced to a frequency.

    Parameters
    ----------
    t0 : astropy.units.Quantity or astropy.time.Time
        Reference time. If a Quantity, assumed to be seconds since the Unix
        epoch.
    offset : astropy.units.Quantity
        Instrumental or processing offset to add.
    f_center : astropy.units.Quantity
        Central observing frequency in MHz.
    DM : astropy.units.Quantity
        Dispersion measure.
    f_ref : astropy.units.Quantity
        Reference frequency in MHz.

    Returns
    -------
    astropy.time.Time
        Time of arrival referred to ``f_ref``.
    """
    shift = K_DM * DM.value * (1 / f_ref.value**2 - 1 / f_center.value**2) * u.s
    if isinstance(t0, Time):
        return t0 + offset + shift
    toa = t0 + offset + shift
    return Time(toa.to_value(u.s), format="unix", scale="utc")


def compute_geometric_delay(t, src, loc1, loc2):
    """Compute the geometric delay between two observatories.

    Parameters
    ----------
    t : astropy.time.Time
        Time of arrival.
    src : astropy.coordinates.SkyCoord
        Source coordinates.
    loc1, loc2 : astropy.coordinates.EarthLocation
        Observatory locations.

    Returns
    -------
    astropy.units.Quantity
        Geometric delay in milliseconds.
    """
    p1 = loc1.get_gcrs(t).cartesian.xyz
    p2 = loc2.get_gcrs(t).cartesian.xyz
    proj = (p2 - p1).dot(src.cartesian.xyz)
    return (proj / const.c).to(u.ms)


def main(dm_opt=550.0, dm_uncertainty=0.2, dsa_mjd=59000.1, 
         chime_unix_timestamp=1598882400.0, source_coord="12:00:00 +20:00:00"):
    """Main analysis function for TOA crossmatching.
    
    Parameters
    ----------
    dm_opt : float, optional
        Optimized dispersion measure in pc/cm^3. Default is 550.0.
    dm_uncertainty : float, optional
        Uncertainty in dispersion measure in pc/cm^3. Default is 0.2.
    dsa_mjd : float, optional
        DSA-110 observation time in MJD. Default is 59000.1.
    chime_unix_timestamp : float, optional
        CHIME observation time in Unix timestamp. Default is 1598882400.0.
    source_coord : str, optional
        Source coordinates in "HH:MM:SS +DD:MM:SS" format. Default is "12:00:00 +20:00:00".
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("--- Analyzing Single Burst ---")

    # ==================================================================
    # This section would contain your CHIME data processing code
    # to derive peak_idx_chime, etc.
    # For this example, we'll use placeholder values.
    # ==================================================================
    DM = dm_opt * (u.pc) / (u.cm**3)
    # Mocking CHIME results
    t0_unix_chime = chime_unix_timestamp * u.s
    offset_chime = 0.01 * u.s

    # CHIME frequency setup
    # Common reference frequency for all TOAs
    F_REF = 400.0 * u.MHz
    # Representative central frequency for CHIME's band (400.39 - 800.39 MHz)
    f_center_chime = 600.39 * u.MHz

    # Your TOA calculation for CHIME
    toa_400_utc_chime = compute_toa(
        t0_unix_chime, offset_chime, f_center_chime, DM, F_REF
    )

    # ==================================================================
    # This section would contain your DSA-110 data processing code
    # ==================================================================
    # Mocking DSA-110 results
    t0_utc_dsa = Time(dsa_mjd, format="mjd", scale="utc")
    offset_dsa = 0.005 * u.s

    # DSA-110 frequency setup
    # Representative central frequency for DSA-110's band (1311.25 - 1498.75 MHz)
    f_center_dsa = 1405.0 * u.MHz

    # Your TOA calculation for DSA-110
    toa_400_utc_dsa = compute_toa(
        t0_utc_dsa, offset_dsa, f_center_dsa, DM, F_REF
    )

    # --- UNCERTAINTY CALCULATION ---
    logging.info("Assumed DM Uncertainty: %.2f pc/cm^3", dm_uncertainty)

    # Calculate timing error for each observatory relative to the 400 MHz reference
    error_chime = calculate_dm_timing_error(dm_uncertainty, f_center_chime, F_REF)
    error_dsa = calculate_dm_timing_error(dm_uncertainty, f_center_dsa, F_REF)

    # The total uncertainty on the offset is the sum in quadrature
    delta_t_uncertainty = np.sqrt(error_chime**2 + error_dsa**2)

    # Format logging with explicit units for clarity
    logging.info("CHIME TOA Error due to DM uncertainty: %.3f %s", 
                error_chime.value, error_chime.unit)
    logging.info("DSA-110 TOA Error due to DM uncertainty: %.3f %s", 
                error_dsa.value, error_dsa.unit)

    # --- Final Results ---
    dt = toa_400_utc_chime - toa_400_utc_dsa
    logging.info("Measured TOA Offset (Δt): %.3f %s", 
                dt.to(u.ms).value, u.ms)
    logging.info("Combined Uncertainty on Δt from DM: ±%.3f %s",
                delta_t_uncertainty.value, delta_t_uncertainty.unit)

    # Geometric delay calculation
    src = SkyCoord(source_coord, unit=(u.hourangle, u.deg), frame="icrs")
    
    geometric_delay = compute_geometric_delay(toa_400_utc_chime, src, CHIME_LOCATION, DSA_LOCATION)
    logging.info("Geometric Delay: %.3f %s", 
                geometric_delay.value, geometric_delay.unit)


if __name__ == "__main__":
    main()
