import numpy as np
from jplephem.spk import SPK
from astropy.time import Time

# Load the ephemeris file
kernel = SPK.open('de440.bsp')
print("✅ JPL Ephemeris DE440 loaded successfully.")

# Get current time
now = Time.now()
julian_date = now.tdb.jd  # use barycentric dynamical time (TDB)

# Earth-Moon barycenter (ID = 3) relative to Solar System Barycenter (ID = 0)
position_vector_km = kernel[0, 3].compute(julian_date)

# Output
print(f"\nTime: {now.iso}")
print(f"Position of Earth-Moon Barycenter relative to SSB (km):")
print(f"X: {position_vector_km[0]:.2f}, Y: {position_vector_km[1]:.2f}, Z: {position_vector_km[2]:.2f}")

# Distance in AU
distance_au = np.linalg.norm(position_vector_km) / 149597870.7
print(f"Current distance from SSB: {distance_au:.2f} AU")
