import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
q_val = 10.0 # Tuned process noise
Q = np.eye(6) * q_val
timing_error_std_sec = 10e-9 
R = np.eye(len(pulsar_vectors)) * (timing_error_std_sec**2)
F = np.eye(6)
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt

# --- MAIN SIMULATION LOOP ---
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
    x = x + K @ y
    I_minus_KH = np.eye(6) - K @ H
    P = I_minus_KH @ P @ I_minus_KH.T + K @ R @ K.T
    estimated_states.append(x)
    # --- PREDICTION STEP ---
    x = F @ x
    P = F @ P @ F.T + Q

print("✅ Simulation complete. Preparing animation...")
true_pos_arr = np.array(true_positions)
est_pos_arr = np.array([state[:3] for state in estimated_states])
time_axis = np.linspace(0, simulation_days, steps)

# --- ANIMATION SETUP ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_title("XNAV EKF Animation", fontsize=16)
ax.set_xlabel("X (millions of km)")
ax.set_ylabel("Y (millions of km)")

# Set plot limits (in millions of km)
lim = orbit_radius_m / 1e9 * 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.grid(True)

# Static objects
sun = plt.Circle((0, 0), 15, color='yellow', zorder=10)
ax.add_artist(sun)

# Animated objects
true_path, = ax.plot([], [], 'b-', label='True Trajectory')
est_path, = ax.plot([], [], 'r--', label='EKF Estimated Trajectory')
true_sc, = ax.plot([], [], 'bo', markersize=8)
est_sc, = ax.plot([], [], 'ro', markersize=8)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='upper right')

# Animation initialization function
def init():
    true_path.set_data([], [])
    est_path.set_data([], [])
    true_sc.set_data([], [])
    est_sc.set_data([], [])
    time_text.set_text('')
    return true_path, est_path, true_sc, est_sc, time_text

# Animation update function (called for each frame)
# (in animate_trajectory.py)

# Animation update function (called for each frame)
def update(frame):
    # Get positions in millions of km for plotting
    x_true = true_pos_arr[:frame+1, 0] / 1e9
    y_true = true_pos_arr[:frame+1, 1] / 1e9
    x_est = est_pos_arr[:frame+1, 0] / 1e9
    y_est = est_pos_arr[:frame+1, 1] / 1e9

    # Update trajectory paths
    true_path.set_data(x_true, y_true)
    est_path.set_data(x_est, y_est)

    # --- FIX IS HERE ---
    # Update spacecraft markers with a list containing the single latest position
    true_sc.set_data([x_true[-1]], [y_true[-1]])
    est_sc.set_data([x_est[-1]], [y_est[-1]])
    # --- END FIX ---

    # Update time text
    time_text.set_text(f'Day: {time_axis[frame]:.1f}')

    return true_path, est_path, true_sc, est_sc, time_text

# Create and run the animation
ani = FuncAnimation(fig, update, frames=len(true_positions),
                    init_func=init, blit=True, interval=50)

plt.show()