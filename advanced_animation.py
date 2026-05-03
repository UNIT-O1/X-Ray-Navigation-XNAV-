import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# --- Part 1: EKF Simulation (Identical to the last working script) ---

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
pulsar_names = list(pulsar_coords.keys())
print(f"✅ Defined {len(pulsar_vectors)} pulsars.")

# --- TRAJECTORY DEFINITION ---
print("🚀 Simulating spacecraft trajectory...")
simulation_days = 365
steps = 150 # More steps for a smoother animation
dt = (simulation_days * 24 * 3600) / steps
orbit_radius_m = 2.2e11

true_positions, true_velocities = [], []
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
x = np.zeros(6); x[:3], x[3:] = true_positions[0], true_velocities[0]
P = np.eye(6) * 1e6
q_val = 10.0; Q = np.eye(6) * q_val
timing_error_std_sec = 10e-9; R = np.eye(len(pulsar_vectors)) * (timing_error_std_sec**2)
F = np.eye(6); F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt

# --- MAIN SIMULATION LOOP ---
estimated_states = []
for i in tqdm(range(steps), desc="Navigating with EKF"):
    true_pos_m = true_positions[i]
    measurements = (pulsar_vectors @ true_pos_m) / c.value
    measurements += np.random.normal(0, timing_error_std_sec, len(pulsar_vectors))
    predicted_measurements = (pulsar_vectors @ x[:3]) / c.value
    H = np.zeros((len(pulsar_vectors), 6)); H[:, :3] = pulsar_vectors / c.value
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
    y = measurements - predicted_measurements
    x = x + K @ y
    I_minus_KH = np.eye(6) - K @ H
    P = I_minus_KH @ P @ I_minus_KH.T + K @ R @ K.T
    estimated_states.append(x)
    x = F @ x; P = F @ P @ F.T + Q

print("✅ Simulation complete. Preparing animation...")

# --- Data Preparation for Plotting ---
true_pos_arr = np.array(true_positions)
est_pos_arr = np.array([state[:3] for state in estimated_states])
errors_km = np.linalg.norm(true_pos_arr - est_pos_arr, axis=1) / 1000.0
time_axis = np.linspace(0, simulation_days, steps)

# --- Part 2: Advanced Animation Setup ---

# Create a figure with a specific layout
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3)
ax_orbit = fig.add_subplot(gs[:, :-1]) # Main orbit plot
ax_sky = fig.add_subplot(gs[0, -1])  # Pulsar sky map
ax_error = fig.add_subplot(gs[1, -1]) # Live error plot
fig.patch.set_facecolor('darkslategrey')

# --- Panel 1: Orbit Animation ---
ax_orbit.set_aspect('equal')
ax_orbit.set_title("XNAV EKF Animation", color='white')
ax_orbit.set_facecolor('black')
ax_orbit.set_xlabel("X (millions of km)", color='white')
ax_orbit.set_ylabel("Y (millions of km)", color='white')
lim = orbit_radius_m / 1e9 * 1.1
ax_orbit.set_xlim(-lim, lim); ax_orbit.set_ylim(-lim, lim)
ax_orbit.tick_params(axis='x', colors='white'); ax_orbit.tick_params(axis='y', colors='white')

# Starry background
stars = np.random.rand(200, 2) * 2 * lim - lim
ax_orbit.scatter(stars[:, 0], stars[:, 1], s=np.random.rand(200) * 2, c='white', alpha=0.5)

# Static Sun
ax_orbit.add_artist(plt.Circle((0, 0), 15, color='#FFD700', zorder=10))

# Animated objects
true_path, = ax_orbit.plot([], [], 'b-', lw=2, label='True Trajectory')
est_path, = ax_orbit.plot([], [], 'r--', lw=2, label='EKF Trajectory')
true_sc = ax_orbit.scatter([], [], s=80, c='blue', marker='^', zorder=11)
est_sc = ax_orbit.scatter([], [], s=80, c='red', marker='^', zorder=11)
time_text = ax_orbit.text(0.02, 0.95, '', transform=ax_orbit.transAxes, color='white', fontsize=12)

# Pulsar beams
beams = [ax_orbit.plot([], [], '-', c='cyan', alpha=0.7, lw=1.5)[0] for _ in pulsar_vectors]
ax_orbit.legend(facecolor='darkgrey')

# --- Panel 2: Pulsar Sky Map ---
ax_sky.set_title("Pulsar Sky Map (RA/Dec)", color='white')
ax_sky.set_facecolor('black')
ax_sky.set_xlabel("Right Ascension (hours)", color='white')
ax_sky.set_ylabel("Declination (degrees)", color='white')
ax_sky.set_xlim(0, 24); ax_sky.set_ylim(-90, 90)
ax_sky.grid(True, linestyle=':', alpha=0.5)
ax_sky.tick_params(axis='x', colors='white'); ax_sky.tick_params(axis='y', colors='white')

for name, coord in pulsar_coords.items():
    ra = coord.ra.hour
    dec = coord.dec.deg
    ax_sky.scatter(ra, dec, s=100, marker='*', c='cyan', label=name)
    ax_sky.text(ra + 0.5, dec, name, color='white', fontsize=9)

# --- Panel 3: Live Error Plot ---
ax_error.set_title("Live Position Error", color='white')
ax_error.set_facecolor('black')
ax_error.set_xlabel("Time (days)", color='white')
ax_error.set_ylabel("Error (km)", color='white')
ax_error.set_xlim(0, simulation_days); ax_error.set_ylim(0, np.max(errors_km) * 1.1)
ax_error.grid(True, linestyle=':', alpha=0.5)
ax_error.tick_params(axis='x', colors='white'); ax_error.tick_params(axis='y', colors='white')
error_line, = ax_error.plot([], [], 'g-', lw=2)

# Animation update function
def update(frame):
    # --- Update Orbit Panel ---
    x_true_km = true_pos_arr[:frame+1, 0] / 1e9
    y_true_km = true_pos_arr[:frame+1, 1] / 1e9
    x_est_km = est_pos_arr[:frame+1, 0] / 1e9
    y_est_km = est_pos_arr[:frame+1, 1] / 1e9
    
    true_path.set_data(x_true_km, y_true_km)
    est_path.set_data(x_est_km, y_est_km)
    
    current_true_pos = np.array([x_true_km[-1], y_true_km[-1]])
    current_est_pos = np.array([x_est_km[-1], y_est_km[-1]])

    true_sc.set_offsets(current_true_pos)
    est_sc.set_offsets(current_est_pos)
    
    # Update pulsar beams to point at the estimated spacecraft
    for i, beam in enumerate(beams):
        pulsar_vec_2d = -pulsar_vectors[i, :2] # Inward pointing vector in XY plane
        pulsar_vec_2d /= np.linalg.norm(pulsar_vec_2d)
        beam_start = current_est_pos + pulsar_vec_2d * 50
        beam.set_data([beam_start[0], current_est_pos[0]], [beam_start[1], current_est_pos[1]])
        # Make beams "flash"
        beam.set_visible(frame % 15 < 3)

    time_text.set_text(f'Day: {time_axis[frame]:.1f}')

    # --- Update Error Panel ---
    error_line.set_data(time_axis[:frame+1], errors_km[:frame+1])
    
    return true_path, est_path, true_sc, est_sc, time_text, error_line, *beams

plt.tight_layout()
ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False) # blit=False is more robust for complex plots
plt.show()