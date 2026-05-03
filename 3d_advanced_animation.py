import numpy as np
from jplephem.spk import SPK
from astropy.constants import c
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# --- Part 1: EKF Simulation (Identical to previous script) ---

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

print("🚀 Simulating spacecraft trajectory...")
simulation_days = 365
steps = 150
dt = (simulation_days * 24 * 3600) / steps
orbit_radius_m = 2.2e11

true_positions, true_velocities = [], []
angles = np.linspace(0, 2 * np.pi, steps)
for angle in angles:
    x = orbit_radius_m * np.cos(angle); y = orbit_radius_m * np.sin(angle); z = 0
    true_positions.append(np.array([x, y, z]))
    speed = 2 * np.pi * orbit_radius_m / (simulation_days * 24 * 3600)
    vx = -speed * np.sin(angle); vy = speed * np.cos(angle); vz = 0
    true_velocities.append(np.array([vx, vy, vz]))

x = np.zeros(6); x[:3], x[3:] = true_positions[0], true_velocities[0]
P = np.eye(6) * 1e12 # Start with higher uncertainty for a better visual effect
q_val = 10.0; Q = np.eye(6) * q_val
timing_error_std_sec = 10e-9; R = np.eye(len(pulsar_vectors)) * (timing_error_std_sec**2)
F = np.eye(6); F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt

estimated_states, covariance_history = [], []
for i in tqdm(range(steps), desc="Navigating with EKF"):
    true_pos_m = true_positions[i]
    measurements = (pulsar_vectors @ true_pos_m) / c.value + np.random.normal(0, timing_error_std_sec, len(pulsar_vectors))
    predicted_measurements = (pulsar_vectors @ x[:3]) / c.value
    H = np.zeros((len(pulsar_vectors), 6)); H[:, :3] = pulsar_vectors / c.value
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.pinv(S)
    y = measurements - predicted_measurements
    x = x + K @ y
    I_minus_KH = np.eye(6) - K @ H
    P = I_minus_KH @ P @ I_minus_KH.T + K @ R @ K.T
    estimated_states.append(x.copy())
    covariance_history.append(P.copy())
    x = F @ x; P = F @ P @ F.T + Q

print("✅ Simulation complete. Preparing 3D animation...")

# --- Data Preparation ---
true_pos_arr_km = np.array(true_positions) / 1000.0
est_states_arr = np.array(estimated_states)
est_pos_arr_km = est_states_arr[:, :3] / 1000.0
est_vel_arr_kms = est_states_arr[:, 3:] / 1000.0
errors_km = np.linalg.norm(true_pos_arr_km - est_pos_arr_km, axis=1)
time_axis = np.linspace(0, simulation_days, steps)

# --- Part 2: Advanced 3D Animation ---

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

lim_km = orbit_radius_m / 1000 * 1.1
ax.set_xlim(-lim_km, lim_km); ax.set_ylim(-lim_km, lim_km); ax.set_zlim(-lim_km, lim_km)
ax.set_xlabel("X (km)", color="white"); ax.set_ylabel("Y (km)", color="white"); ax.set_zlabel("Z (km)", color="white")
ax.tick_params(axis='x', colors='grey'); ax.tick_params(axis='y', colors='grey'); ax.tick_params(axis='z', colors='grey')
ax.grid(color='grey', linestyle=':', linewidth=0.5)

# --- Static Objects ---
# Sun
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sun = np.cos(u)*np.sin(v)*lim_km*0.05
y_sun = np.sin(u)*np.sin(v)*lim_km*0.05
z_sun = np.cos(v)*lim_km*0.05
ax.plot_surface(x_sun, y_sun, z_sun, color="yellow")

# Ecliptic Plane
plane_radius = lim_km
plane_x = np.linspace(-plane_radius, plane_radius, 10)
plane_y = np.linspace(-plane_radius, plane_radius, 10)
plane_x, plane_y = np.meshgrid(plane_x, plane_y)
plane_z = np.zeros_like(plane_x)
ax.plot_surface(plane_x, plane_y, plane_z, color='blue', alpha=0.1)

# Pulsars on a Celestial Sphere
sphere_radius = lim_km * 0.9
pulsar_pos_3d = pulsar_vectors * sphere_radius
ax.scatter(pulsar_pos_3d[:,0], pulsar_pos_3d[:,1], pulsar_pos_3d[:,2], s=100, c='cyan', marker='*')
for i, name in enumerate(pulsar_names):
    ax.text(pulsar_pos_3d[i,0]*1.05, pulsar_pos_3d[i,1]*1.05, pulsar_pos_3d[i,2]*1.05, name, color='cyan', fontsize=10)

# --- Animated Artists ---
true_path, = ax.plot([], [], [], 'b-', lw=2, label='True Trajectory')
est_path, = ax.plot([], [], [], 'r--', lw=2, label='EKF Trajectory')
true_sc = ax.scatter([], [], [], s=80, c='blue', marker='o')
est_sc = ax.scatter([], [], [], s=80, c='red', marker='o')
beams = [ax.plot([], [], [], '-', c='cyan', alpha=0.5, lw=1)[0] for _ in pulsar_vectors]
vel_vector = ax.quiver(0, 0, 0, 0, 0, 0, color='lime', length=lim_km*0.2, normalize=True)
info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, color='white', fontsize=12, verticalalignment='top')
ax.legend(facecolor='darkgrey')

# Helper to plot the covariance ellipsoid
cov_ellipsoid = None
def plot_covariance_ellipsoid(ax, P_pos, center, color, alpha):
    global cov_ellipsoid
    if cov_ellipsoid:
        cov_ellipsoid.remove()
    
    # Find eigenvalues and eigenvectors for orientation and size
    eigvals, eigvecs = np.linalg.eigh(P_pos)
    radii = np.sqrt(eigvals) * 3 # 3-sigma confidence
    
    u = np.linspace(0.0, 2.0 * np.pi, 10)
    v = np.linspace(0.0, np.pi, 10)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(eigvecs, [x[i, j], y[i, j], z[i, j]]) + center

    cov_ellipsoid = ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2)

# Animation update function
def update(frame):
    # Update paths
    true_path.set_data_3d(true_pos_arr_km[:frame+1, 0], true_pos_arr_km[:frame+1, 1], true_pos_arr_km[:frame+1, 2])
    est_path.set_data_3d(est_pos_arr_km[:frame+1, 0], est_pos_arr_km[:frame+1, 1], est_pos_arr_km[:frame+1, 2])
    
    # Update spacecraft markers
    current_true_pos = true_pos_arr_km[frame]
    current_est_pos = est_pos_arr_km[frame]
    true_sc._offsets3d = ([current_true_pos[0]], [current_true_pos[1]], [current_true_pos[2]])
    est_sc._offsets3d = ([current_est_pos[0]], [current_est_pos[1]], [current_est_pos[2]])

    # Update pulsar beams
    for i, beam in enumerate(beams):
        beam.set_data_3d([pulsar_pos_3d[i,0], current_est_pos[0]], 
                        [pulsar_pos_3d[i,1], current_est_pos[1]], 
                        [pulsar_pos_3d[i,2], current_est_pos[2]])
        beam.set_visible(frame % 20 < 4) # Flash the beams

    # Update velocity vector
    global vel_vector
    vel_vector.remove()
    current_vel = est_vel_arr_kms[frame]
    vel_vector = ax.quiver(current_est_pos[0], current_est_pos[1], current_est_pos[2],
                           current_vel[0], current_vel[1], current_vel[2],
                           color='lime', length=lim_km*0.2, normalize=True)

    # Update covariance ellipsoid
    P_pos = covariance_history[frame][:3, :3]
    plot_covariance_ellipsoid(ax, P_pos, current_est_pos, 'fuchsia', 0.2)

    # Update info text
    speed = np.linalg.norm(current_vel)
    info_text.set_text(f'Day: {time_axis[frame]:.1f}\n'
                       f'Position Error: {errors_km[frame]:,.0f} km\n'
                       f'Speed: {speed:,.0f} km/s')

    return true_path, est_path, true_sc, est_sc, vel_vector, info_text, cov_ellipsoid, *beams

ax.view_init(elev=25., azim=30)
ani = FuncAnimation(fig, update, frames=steps, interval=40, blit=False)
plt.show()