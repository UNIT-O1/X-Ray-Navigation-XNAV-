from astropy.time import Time
import astropy.units as u

# Create a time object for the current time in the standard UTC scale
t_utc = Time.now()
print(f"Spacecraft Clock Time (UTC): {t_utc.iso}")

# Convert it to Barycentric Dynamical Time (TDB)
# TDB is a uniform time scale corrected for relativistic effects at the SSB.
t_tdb = t_utc.tdb
print(f"Barycentric Time (TDB):      {t_tdb.iso}")

# Let's see the difference between the two scales
time_difference = (t_tdb - t_utc).to(u.millisecond)
print(f"\nDifference (TDB - UTC): {time_difference:.4f}")