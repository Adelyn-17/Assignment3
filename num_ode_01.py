import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

print("numpy版本:", np.__version__)
print("matplotlib版本:", matplotlib.__version__)

plt.show(block=True)

t0 = 0.0  
t_end = 5.0 
h = 0.1  
y0 = 1.0  


# 导函数 f(t, y) = y'
def f(t, y):
    """ODE: y' = t^2 - y"""
    return t ** 2 - y


# 精确解 y(t) = -e^(-t) + t^2 - 2t + 2
def exact_solution(t):
    """Exact solution for the ODE"""
    return -np.exp(-t) + t ** 2 - 2 * t + 2


t_points = np.arange(t0, t_end + h, h)
N = len(t_points)

y_euler = np.zeros(N)
y_heun = np.zeros(N)
y_rk4 = np.zeros(N)

y_euler[0] = y0
y_heun[0] = y0
y_rk4[0] = y0


# 1. Euler's Method
# y_{i+1} = y_i + h * f(t_i, y_i)
for i in range(N - 1):
    y_euler[i + 1] = y_euler[i] + h * f(t_points[i], y_euler[i])

# 2. Heun's Method
# Predictor: y^* = y_i + h * f(t_i, y_i)
# Corrector: y_{i+1} = y_i + (h/2) * [f(t_i, y_i) + f(t_{i+1}, y^*)]
for i in range(N - 1):
    t_i = t_points[i]
    y_i = y_heun[i]


    y_star = y_i + h * f(t_i, y_i)

    t_next = t_points[i + 1]
    y_heun[i + 1] = y_i + (h / 2.0) * (f(t_i, y_i) + f(t_next, y_star))

# 3.RK-4 Method
# y_{i+1} = y_i + (h/6) * (k1 + 2k2 + 2k3 + k4)
for i in range(N - 1):
    t_i = t_points[i]
    y_i = y_rk4[i]

    k1 = f(t_i, y_i)
    k2 = f(t_i + h / 2, y_i + (h / 2) * k1)
    k3 = f(t_i + h / 2, y_i + (h / 2) * k2)
    k4 = f(t_i + h, y_i + h * k3)

    y_rk4[i + 1] = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

y_exact = exact_solution(t_points)
error_euler = np.abs(y_exact - y_euler)
error_heun = np.abs(y_exact - y_heun)
error_rk4 = np.abs(y_exact - y_rk4)

# 1. Trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_points, y_exact, label='Exact Solution', color='black', linewidth=3, linestyle='--')
plt.plot(t_points, y_euler, label='Euler Method (h=0.1)', linestyle='-', marker='o', markersize=3)
plt.plot(t_points, y_heun, label="Heun's Method (h=0.1)", linestyle='-', marker='x', markersize=4)
plt.plot(t_points, y_rk4, label='RK-4 Method (h=0.1)', linestyle='-', marker='s', markersize=3)

plt.title(r'ODE Trajectories: $y\' = t^2 - y$ with $y(0)=1$')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)

# 2.Errors
plt.figure(figsize=(10, 6))
plt.plot(t_points, error_euler, label='Euler Error', linestyle='-', marker='o', markersize=3)
plt.plot(t_points, error_heun, label="Heun's Error", linestyle='-', marker='x', markersize=4)
plt.plot(t_points, error_rk4, label='RK-4 Error', linestyle='-', marker='s', markersize=3)

plt.title(r'Absolute Errors ( $|y_{exact} - y_{approx}|$ )')
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.yscale('log') 
plt.legend()
plt.grid(True)

plt.show(block=True)
