import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- SETUP ---
kernel = SPK.open('de440.bsp')
print("✅ JPL Ephemeris DE440 loaded.")

pulsar_coords = {
    "J0437-4715": SkyCoord("04h37m15.8s", "-47d15m09s", frame='icrs'),
    "J1939+2134": SkyCoord("19h39m38.5s", "+21d34m59s", frame='icrs'),
    "J1713+0747": SkyCoord("17h13m49.5s", "+07d47m37s", frame='icrs'),
    "J0218+4232": SkyCoord("02h18m16.1s", "+42d32m17s", frame='icrs'),
}
pulsar_vectors = {name: coord.cartesian.xyz.value for name, coord in pulsar_coords.items()}
pulsars = list(pulsar_vectors.values())
print(f"✅ Defined {len(pulsars)} pulsars.")

# --- TRAJECTORY DEFINITION ---
print("🚀 Simulating spacecraft trajectory...")
simulation_days = 365
steps = 100
orbit_radius_km = 2.2e8
angles = np.linspace(0, 2 * np.pi, steps)
true_positions = []
for angle in angles:
    x = orbit_radius_km * np.cos(angle)
    y = orbit_radius_km * np.sin(angle)
    z = 0
    true_positions.append(np.array([x, y, z]))

# --- SOLVER FUNCTION ---
def calculate_residuals(r_guess_millions_km, observed_delays):
    r_guess_km = r_guess_millions_km * 1e6
    residuals = []
    for i, n_hat in enumerate(pulsars):
        calculated_delay = np.dot(r_guess_km * 1000.0, n_hat) / c.value
        residuals.append(calculated_delay - observed_delays[i])
    return np.array(residuals)

# --- MAIN SIMULATION LOOP (Simplified Logic) ---
solved_positions = []

# Initialize the guess at the known starting point of the orbit.
initial_guess_millions_km = true_positions[0] / 1e6

for r_sc_true_km in tqdm(true_positions, desc="Navigating"):
    # 1. Simulate observation
    observed_delays = []
    for n_hat in pulsars:
        delay = np.dot(r_sc_true_km * 1000.0, n_hat) / c.value
        observed_delays.append(delay)
    observed_delays += np.random.normal(0, 10e-9, len(pulsars))

    # 2. Solve for the current position using the last known guess
    result = least_squares(calculate_residuals, initial_guess_millions_km, args=([observed_delays]))
    r_sc_solved_km = result.x * 1e6
    solved_positions.append(r_sc_solved_km)
    
    # 3. Update the guess for the next step with the result from this step.
    initial_guess_millions_km = result.x

# --- ANALYSIS AND PLOTTING ---
print("✅ Simulation complete. Generating plots...")
true_pos = np.array(true_positions)
solved_pos = np.array(solved_positions)

errors_km = np.linalg.norm(true_pos - solved_pos, axis=1)
time_axis = np.linspace(0, simulation_days, steps)

# PLOT 1: 3D Trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("XNAV Trajectory Reconstruction", fontsize=16)
ax.plot(true_pos[:,0]/1e6, true_pos[:,1]/1e6, true_pos[:,2]/1e6, 'b-', label='True Trajectory', linewidth=2)
ax.plot(solved_pos[:,0]/1e6, solved_pos[:,1]/1e6, solved_pos[:,2]/1e6, 'r.--', label='XNAV Solved Trajectory')
ax.set_xlabel("X (millions of km)")
ax.set_ylabel("Y (millions of km)")
ax.set_zlabel("Z (millions of km)")
ax.legend()
plt.tight_layout()
plt.show()


# PLOT 2: Positioning Error Over Time
plt.figure(figsize=(12, 6))
plt.title("Positioning Error Over Time", fontsize=16)
plt.plot(time_axis, errors_km, 'g-')
plt.xlabel("Time (days)")
plt.ylabel("Position Error (km)")
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()