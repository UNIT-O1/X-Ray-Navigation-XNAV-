import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
from scipy.optimize import least_squares

# --- SETUP (Same as before) ---
kernel = SPK.open('de440.bsp')
pulsar_coords = {
    "J0437-4715": SkyCoord("04h37m15.8s", "-47d15m09s", frame='icrs'),
    "J1939+2134": SkyCoord("19h39m38.5s", "+21d34m59s", frame='icrs'),
    "J1713+0747": SkyCoord("17h13m49.5s", "+07d47m37s", frame='icrs'),
    "J0218+4232": SkyCoord("02h18m16.1s", "+42d32m17s", frame='icrs'),
}
pulsar_vectors = {name: coord.cartesian.xyz.value for name, coord in pulsar_coords.items()}
pulsars = list(pulsar_vectors.values())

# --- SOLVER FUNCTION (Same as before) ---
def calculate_residuals(r_guess_millions_km, observed_delays):
    r_guess_km = r_guess_millions_km * 1e6
    residuals = []
    for i, n_hat in enumerate(pulsars):
        calculated_delay = np.dot(r_guess_km * 1000.0, n_hat) / c.value
        residuals.append(calculated_delay - observed_delays[i])
    return np.array(residuals)

# --- SINGLE POINT TEST ---
print("--- Running Single Point Debug Test ---")
orbit_radius_km = 2.2e8

# Let's test the 50th step (halfway through the orbit)
step_index = 50
total_steps = 100

# This is the "previous" position (step 49)
previous_angle = (step_index - 1) * (2 * np.pi / total_steps)
x_prev = orbit_radius_km * np.cos(previous_angle)
y_prev = orbit_radius_km * np.sin(previous_angle)
previous_true_pos_km = np.array([x_prev, y_prev, 0.0])

# This is the "current" true position (step 50)
current_angle = step_index * (2 * np.pi / total_steps)
x_curr = orbit_radius_km * np.cos(current_angle)
y_curr = orbit_radius_km * np.sin(current_angle)
r_sc_true_km = np.array([x_curr, y_curr, 0.0])

# This is our smart initial guess
initial_guess_millions_km = previous_true_pos_km / 1e6

# Simulate the observation at the current position
observed_delays = []
for n_hat in pulsars:
    delay = np.dot(r_sc_true_km * 1000.0, n_hat) / c.value
    observed_delays.append(delay)
observed_delays += np.random.normal(0, 10e-9, len(pulsars))

# Run the solver just once
result = least_squares(calculate_residuals, initial_guess_millions_km, args=([observed_delays]))
r_sc_solved_km = result.x * 1e6

# --- RESULTS ---
error_km = np.linalg.norm(r_sc_solved_km - r_sc_true_km)

print(f"\nInitial Guess (km): {np.round(previous_true_pos_km, 0)}")
print(f"True Position (km):   {np.round(r_sc_true_km, 0)}")
print(f"Solved Position (km): {np.round(r_sc_solved_km, 0)}")
print(f"\n✅ Final Error for this single point: {error_km:.2f} km")