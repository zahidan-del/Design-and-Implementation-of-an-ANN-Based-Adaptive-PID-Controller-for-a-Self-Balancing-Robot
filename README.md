# ANN-Based Adaptive PID Controller for a Self-Balancing Robot

This repository contains the implementation of an **Artificial Neural Network (ANN)-based Adaptive PID Controller** for a two-wheeled self-balancing robot. The system is designed to improve stability and robustness by dynamically adjusting PID gains (**Kp, Ki, Kd**) in real time, based on the robot's current state. The ANN model is trained offline and deployed on an embedded platform using **ONNX Runtime** for efficient inference.

---

## Key Features
- **Self-balancing control** using ANN-based adaptive PID.
- Real-time adjustment of PID gains (**Kp, Ki, Kd**) based on robot state.
- Embedded implementation on **Raspberry Pi 3 Model B+**.
- **MPU6050 IMU sensor** for angle and angular velocity measurement.
- Complementary filter for accurate and stable angle estimation.
- Comparative analysis between **conventional PID** and **ANN-Adaptive PID**.
- Deployed ANN model in **ONNX** format for efficient runtime performance.

---

## Technologies Used
- **Python** – Control algorithms, data logging, and ANN implementation.
- **PyTorch** – Training the ANN model.
- **ONNX Runtime** – Running the trained ANN model on Raspberry Pi.
- **PID Controller** – Baseline comparison with adaptive method.
- **Complementary Filter** – Fusing accelerometer and gyroscope data.
- **Matplotlib / NumPy** – Data analysis and visualization.

---

## Main Components Used
- **Raspberry Pi 3 Model B+** – Main processing unit for control algorithms and ANN inference.
- **MPU6050 IMU Sensor** – Provides pitch angle and angular velocity data.
- **JGA25-370 DC Motors (x2)** – Drive the wheels for balancing.
- **L298N Motor Driver** – Motor control interface.
- **3S Li-ion 18650 Battery Pack (11.1V)** – Power source.
- **Step-down Converter (5V 3A)** – Stable power supply for Raspberry Pi.

---

## Performance Summary
| Metric                     | Conventional PID | ANN-Adaptive PID |
|----------------------------|------------------|------------------|
| Rise Time (s)              | 0.0170           | 0.0175           |
| Overshoot / Undershoot (%) | -114.35          | -75.59           |
| Peak Time (s)              | 2.72             | 4.40             |
| Steady-State Error (°)     | +0.3763          | -0.2381          |

The ANN-Adaptive PID shows **~34% improvement** in overshoot reduction and better damping performance compared to the conventional PID controller.

---

## Future Improvements
- Expand the ANN training dataset for better generalization.
- Implement online learning or reinforcement learning for continuous adaptation.
- Test with varied real-world disturbances such as uneven terrain and physical pushes.

---

## Contact
📧 Email: zahidan54@gmail.com  
📷 Instagram: [@zhdnakhmad](https://instagram.com/zhdnakhmad)
