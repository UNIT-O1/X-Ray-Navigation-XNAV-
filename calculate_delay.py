import numpy as np
from jplephem.spk import SPK
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.constants import c

# --- SETUP ---
# Load the ephemeris (make sure 'de440.bsp' is in your folder)
kernel = SPK.open('de440.bsp')
print("✅ JPL Ephemeris DE440 loaded.")

# Define our target pulsar: PSR B1937+21
pulsar = SkyCoord("19h39m38.56s", "+21d34m59.1s", frame='icrs')
print(f"✅ Pulsar defined: {pulsar}")

# --- CALCULATION ---
# 1. Get the spacecraft's position vector (we'll use Earth's position)
time_now = Time.now()
julian_date = time_now.tdb.jd  # SPK kernels expect TDB (not UTC JD)

# Earth-Moon barycenter (3) relative to Solar System Barycenter (0)
r_sc_km = kernel[0, 3].compute(julian_date)  
r_sc_m = r_sc_km * 1000.0  # convert to meters

# 2. Get the unit vector pointing to the pulsar
n_hat = pulsar.cartesian.xyz.value  # already unit length in ICRS

# 3. Calculate the dot product (projected distance along pulsar direction)
projected_distance = np.dot(r_sc_m, n_hat)

# 4. Time delay = projected distance / speed of light
time_delay_seconds = projected_distance / c.value

# --- OUTPUT ---
print(f"\n--- Results for {time_now.iso} ---")
print(f"Spacecraft position (X, Y, Z) from SSB: {r_sc_km.round(0)} km")
print(f"Pulsar unit vector (nx, ny, nz): {n_hat.round(3)}")
print(f"Projected distance along pulsar direction: {projected_distance / 1e9:.2f} million km")
print(f"✅ Calculated Geometric Time Delay: {time_delay_seconds:.4f} seconds "
      f"({time_delay_seconds * 1000:.2f} ms)")
