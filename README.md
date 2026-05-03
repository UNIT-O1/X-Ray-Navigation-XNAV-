# XNAV: Deep Space Autonomous Navigation

> Sub-kilometer autonomous positioning beyond Earth orbit using millisecond pulsar timing and numerically stable state estimation.

---

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Astropy](https://img.shields.io/badge/Astropy-Library-orange?style=flat-square)
![Three.js](https://img.shields.io/badge/Three.js-Visualizer-000000?style=flat-square&logo=three.js)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)

---

## 🚀 Overview

XNAV (X-ray Pulsar Navigation) implements a fully autonomous navigation framework for deep-space missions, eliminating reliance on Earth-based tracking systems.

Traditional spacecraft navigation depends on the Deep Space Network (DSN), which introduces latency, limited coverage, and scalability constraints. XNAV replaces this paradigm with a decentralized, physics-grounded “Galactic GPS”, using naturally occurring millisecond pulsars as ultra-stable beacons.

By measuring pulse arrival times and solving for spacecraft position relative to the Solar System Barycenter (SSB), the system performs real-time onboard navigation—critical for future interplanetary and deep-space autonomy.

---

## ✨ Features

- Autonomous state estimation using Extended Kalman Filter (EKF) with Joseph-form covariance updates for long-term numerical stability  
- Pulsar-based triangulation using timing residuals from millisecond pulsars  
- High-precision celestial modeling via JPL DE440 ephemeris integration  
- Optimization-based correction using Gauss-Newton and least-squares solvers  
- Multi-modal visualization with scientific 3D simulations and interactive web interface  

---

## 🧱 Tech Stack

- Core: Python 3.9+, NumPy, SciPy  
- Astrodynamics: Astropy, jplephem  
- Visualization: Matplotlib (3D), Three.js, Tailwind CSS  
- Utilities: tqdm  

---

## ⚙️ Architecture

1. Environment Initialization  
   - Loads JPL DE440 ephemeris data  
   - Defines pulsar vectors in ICRS frame  

2. Observation Model  
   - Computes geometric delay: Δt = (r · n̂) / c  
   - Measures timing differences between expected and observed signals  

3. State Estimation  
   - Prediction: propagates spacecraft state using orbital dynamics  
   - Correction: updates state via EKF or Gauss-Newton optimization  

4. Error Analysis & Visualization  
   - Tracks divergence between true and estimated trajectory  
   - Visualizes uncertainty through covariance evolution  

---

## 🌌 Why This Matters

Deep-space missions today are fundamentally Earth-dependent.

The Deep Space Network introduces communication delays, limited scalability, and continuous dependence on ground infrastructure. As missions extend farther, this model becomes increasingly inefficient.

XNAV demonstrates a shift toward onboard intelligence and decentralized navigation, enabling spacecraft to determine their position independently using astrophysical signals.

This approach is essential for:
- Mars and outer planet missions  
- Autonomous deep-space probes  
- Scalable interplanetary exploration  

---

## 📡 Realism & Scientific Grounding

XNAV is grounded in real navigation and astrophysical principles.

Pulsar-based navigation is actively researched and validated in modern space systems. The use of JPL ephemeris ensures astronomical accuracy, while Kalman filtering reflects real spacecraft estimation pipelines.

What is realistic:
- Pulsar timing-based positioning  
- Barycentric coordinate modeling  
- Kalman filter-based navigation  
- Timing residual-based correction  

What is simplified:
- Sensor noise and detector limitations  
- Full relativistic corrections  
- Environmental perturbations  

This makes XNAV a high-fidelity conceptual prototype of real autonomous navigation systems.

---

## 📸 Preview

- 3D orbital trajectories  
- Pulsar beam visualization  
- Covariance ellipsoids (uncertainty modeling)  

(Add screenshots or demo GIF here)

---

## 🛠️ Setup

1. Create environment  

python -m venv xnav_env
source xnav_env/bin/activate # Windows: xnav_env\Scripts\activate


2. Install dependencies  

pip install numpy astropy jplephem matplotlib scipy tqdm


3. Data setup  
- Download JPL DE440 ephemeris file (de440.bsp)  
- Place it in the project root directory  

---

## ▶️ Usage

Scientific 3D Simulation  

python 3d_advanced_animation.py


Mission Control Dashboard  

python advanced_animation.py


Basic Trajectory Simulation  

python animate_trajectory.py


Interactive Web Interface  
Open `XNAV.html` in a browser  

---

## 💡 Highlights

- Numerically stable EKF using Joseph-form covariance updates  
- Fully physics-driven navigation system  
- Zero dependence on Earth-based tracking infrastructure  
- Bridges astrophysics with real navigation system design  

---

## 🔮 Roadmap

- Relativistic corrections (Shapiro delay, aberration)  
- Solar radiation pressure and perturbation modeling  
- Multi-spacecraft cooperative navigation  
- Sensor fusion with onboard instrumentation  

---

## 👤 Author

Dhruv Vaghela  
B.Tech Mathematics & Computing, NIT Warangal  

---

XNAV represents a step toward fully autonomous deep-space systems—where spacecraft navigate u
