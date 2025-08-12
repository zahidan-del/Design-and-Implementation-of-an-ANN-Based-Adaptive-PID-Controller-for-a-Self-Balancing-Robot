import RPi.GPIO as GPIO
import time
import math
from mpu6050 import mpu6050
import onnxruntime as ort
import numpy as np
import csv
import datetime

# =================== Logging Setup ===================
# Create a CSV file with timestamped filename to log data
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"balancing_log_{timestamp}.csv"
log_file = open(log_filename, mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow([
    "Time", "Pitch", "Error", "Integral_Error", "Angular_Velocity",
    "Integral_Angular_Error", "Position", "Velocity", "Kp", "Ki", "Kd"
])

# =================== MPU6050 Sensor Initialization ===================
sensor = mpu6050(0x68)

# =================== GPIO Setup ===================
GPIO.setmode(GPIO.BCM)
IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 23
ENA = 18  # PWM0
ENB = 13  # PWM1

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Setup PWM for both motors
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

# =================== Load ONNX Model ===================
onnx_model_path = "balancing_nn.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =================== Scaling Parameters ===================
# Input scaling min/max values
X_min = np.array([-45.413, -185.87, -24.537, -24.537, -185.87], dtype=np.float32)
X_max = np.array([13.692, 224.725, -7.969, -7.969, 224.725], dtype=np.float32)

def scale_input(x):
    """Scale the input vector to [0,1] range."""
    denom = X_max - X_min
    denom[denom == 0] = 1  # avoid division by zero
    return (x - X_min) / denom

# Output scaling min/max values
y_min = np.array([13.8, 0.92, 11.5])  
y_max = np.array([13.9, 0.921, 11.7])         

def unscale_output(y_scaled):
    """Unscale the model output back to original Kp, Ki, Kd values."""
    return y_scaled * (y_max - y_min) + y_min

# =================== Variables ===================
target_angle = -2.745
integral = 0
last_error = 0

alpha = 0.95  # Complementary filter constant
angle = 0
last_time = time.time()
start_time = time.time()

angular_velocity = 0
integral_angular_error = 0
position = 0
velocity = 0

# =================== Functions ===================
def get_filtered_angle():
    """
    Calculate filtered pitch angle using a complementary filter.
    Combines accelerometer and gyroscope data from MPU6050.
    """
    global angle, last_time, angular_velocity
    now = time.time()
    dt = now - last_time
    last_time = now

    accel = sensor.get_accel_data()
    gyro = sensor.get_gyro_data()

    try:
        accel_angle = math.degrees(math.atan2(accel['y'], accel['z']))
    except:
        accel_angle = 0

    gyro_rate = gyro['x']
    angular_velocity = gyro_rate
    angle = alpha * (angle + gyro_rate * dt) + (1 - alpha) * accel_angle
    return angle, dt

def set_motor(speed):
    """
    Control motor direction and speed based on the calculated PID output.
    Speed range: -100 to 100.
    """
    speed = max(min(speed, 100), -100)
    if speed > 0:
        GPIO.output(IN1, True)
        GPIO.output(IN2, False)
        GPIO.output(IN3, True)
        GPIO.output(IN4, False)
    elif speed < 0:
        GPIO.output(IN1, False)
        GPIO.output(IN2, True)
        GPIO.output(IN3, False)
        GPIO.output(IN4, True)
    else:
        GPIO.output(IN1, False)
        GPIO.output(IN2, False)
        GPIO.output(IN3, False)
        GPIO.output(IN4, False)

    pwm_val = abs(speed)
    pwmA.ChangeDutyCycle(pwm_val)
    pwmB.ChangeDutyCycle(pwm_val)

# =================== Main Loop ===================
try:
    print("Starting balancing robot using ONNX model...")
    time.sleep(2)

    while True:
        # Get filtered pitch angle
        pitch, dt = get_filtered_angle()

        # PID-related error calculations
        error = pitch - target_angle
        integral += error
        derivative = error - last_error

        # Additional features for NN input
        integral_angular_error += angular_velocity * dt
        velocity = angular_velocity
        position += velocity * dt

        # Prepare model input
        raw_input = np.array([integral, angular_velocity, integral_angular_error, position, velocity], dtype=np.float32)
        scaled_input = scale_input(raw_input)
        model_input = scaled_input.reshape(1, -1)

        # Run inference on ONNX model
        outputs = session.run([output_name], {input_name: model_input})
        kp_scaled, ki_scaled, kd_scaled = outputs[0][0]

        # Unscale the output back to Kp, Ki, Kd
        kp, ki, kd = unscale_output(np.array([kp_scaled, ki_scaled, kd_scaled]))

        # PID control output
        output = kp * error + ki * integral + kd * derivative
        last_error = error

        # Relative time since start
        current_time = time.time() - start_time

        # Print values to console
        print(f"Pitch: {pitch:.2f}° | Error: {error:.2f} | Kp: {kp:.3f} | Ki: {ki:.5f} | Kd: {kd:.3f} | Output: {output:.2f}")

        # Log data to CSV
        csv_writer.writerow([
            round(current_time, 4), round(pitch, 2), round(error, 2), round(integral, 4),
            round(angular_velocity, 4), round(integral_angular_error, 4),
            round(position, 4), round(velocity, 4), round(kp, 4), round(ki, 6), round(kd, 4)
        ])

        # Apply motor control
        set_motor(-output)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    log_file.close()
    print(f"Data saved to: {log_filename}")
