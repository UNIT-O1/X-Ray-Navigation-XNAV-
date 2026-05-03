import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
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
pulsar_vectors = np.array([coord.cartesian.xyz.value for coord in pulsar_coords.values()])
print(f"✅ Defined {len(pulsar_vectors)} pulsars.")

# --- TRAJECTORY DEFINITION ---
print("🚀 Simulating spacecraft trajectory...")
simulation_days = 365
steps = 100
dt = (simulation_days * 24 * 3600) / steps
orbit_radius_m = 2.2e11

true_positions = []
true_velocities = []
angles = np.linspace(0, 2 * np.pi, steps)
for angle in angles:
    x = orbit_radius_m * np.cos(angle)
    y = orbit_radius_m * np.sin(angle)
    z = 0
    true_positions.append(np.array([x, y, z]))
    speed = 2 * np.pi * orbit_radius_m / (simulation_days * 24 * 3600)
    vx = -speed * np.sin(angle)
    vy = speed * np.cos(angle)
    vz = 0
    true_velocities.append(np.array([vx, vy, vz]))

# --- EKF INITIALIZATION ---
x = np.zeros(6) 
x[:3] = true_positions[0]
x[3:] = true_velocities[0]
P = np.eye(6) * 1e6
q_val = 10.0
Q = np.eye(6) * q_val
timing_error_std_sec = 10e-9 
R = np.eye(len(pulsar_vectors)) * (timing_error_std_sec**2)
F = np.eye(6)
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt

# --- MAIN SIMULATION LOOP (With Stability Fix) ---
estimated_states = []

for i in tqdm(range(steps), desc="Navigating with EKF"):
    # --- UPDATE STEP ---
    true_pos_m = true_positions[i]
    measurements = (pulsar_vectors @ true_pos_m) / c.value
    measurements += np.random.normal(0, timing_error_std_sec, len(pulsar_vectors))
    
    predicted_measurements = (pulsar_vectors @ x[:3]) / c.value
    H = np.zeros((len(pulsar_vectors), 6))
    H[:, :3] = pulsar_vectors / c.value
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = measurements - predicted_measurements
    
    # Update state estimate
    x = x + K @ y
    
    # ***************************************************************
    # THE FIX: Update Covariance using the stable Joseph Form
    I_minus_KH = np.eye(6) - K @ H
    P = I_minus_KH @ P @ I_minus_KH.T + K @ R @ K.T
    # ***************************************************************
    
    estimated_states.append(x)

    # --- PREDICTION STEP ---
    x = F @ x
    P = F @ P @ F.T + Q

# --- ANALYSIS AND PLOTTING ---
print("✅ Simulation complete. Generating plots...")
true_pos_arr = np.array(true_positions)
est_pos_arr = np.array([state[:3] for state in estimated_states])
errors_km = np.linalg.norm(true_pos_arr - est_pos_arr, axis=1) / 1000.0
time_axis = np.linspace(0, simulation_days, steps)

# PLOT 1: 3D Trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("EKF Trajectory Reconstruction (Stable)", fontsize=16)
ax.plot(true_pos_arr[:,0]/1e9, true_pos_arr[:,1]/1e9, true_pos_arr[:,2]/1e9, 'b-', label='True Trajectory', linewidth=2)
ax.plot(est_pos_arr[:,0]/1e9, est_pos_arr[:,1]/1e9, est_pos_arr[:,2]/1e9, 'r.--', label='EKF Estimated Trajectory')
ax.set_xlabel("X (millions of km)")
ax.set_ylabel("Y (millions of km)")
ax.set_zlabel("Z (millions of km)")
ax.legend()
ax.view_init(elev=30., azim=45)
plt.tight_layout()
plt.show()

# PLOT 2: Positioning Error Over Time
plt.figure(figsize=(12, 6))
plt.title("Positioning Error Over Time (EKF Stable)", fontsize=16)
plt.plot(time_axis, errors_km, 'g-')
plt.xlabel("Time (days)")
plt.ylabel("Position Error (km)")
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()