import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

t0, t_end = 1.0, 3.0
x_start = 1.0      
x_target = -1.0    
h = 0.05

def system_4_1(t, U):
    u1, u2 = U
    du1dt = u2
    du2dt = (-2.0/t)*u2 + (2.0/t**2)*u1 + (10.0 * np.cos(np.log(t))) / (t**2)
    return np.array([du1dt, du2dt])

def rk4_solve(f, t_span, u0, h):
    ts = np.arange(t_span[0], t_span[1] + h, h)
    if ts[-1] > t_span[1]: ts = ts[:-1] 
    us = np.zeros((len(ts), len(u0)))
    us[0] = u0
    for i in range(len(ts) - 1):
        k1 = f(ts[i], us[i])
        k2 = f(ts[i] + 0.5*h, us[i] + 0.5*h*k1)
        k3 = f(ts[i] + 0.5*h, us[i] + 0.5*h*k2)
        k4 = f(ts[i] + h, us[i] + h*k3)
        us[i+1] = us[i] + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return ts, us

def objective(s):
    u0 = [x_start, float(s)]
    ts, us = rk4_solve(system_4_1, [t0, t_end], u0, h)
    return us[-1, 0] - x_target

correct_s = fsolve(objective, 0.0)[0]
print(f"经过打靶法找到的初始斜率 x'(1) = {correct_s:.6f}")

t_num, u_num = rk4_solve(system_4_1, [t0, t_end], [x_start, correct_s], h)
x_num = u_num[:, 0]

def exact_solution(t):
    numerator = (4.335950689
                 - 0.3359506908 * t**3
                 - 3.0 * t**2 * np.cos(np.log(t))
                 + t**2 * np.sin(np.log(t)))
    return numerator / (t**2)

t_fine = np.linspace(t0, t_end, 500)
x_exact = exact_solution(t_fine)

plt.figure(figsize=(10, 6))
plt.plot(t_num, x_num, 'ro', label=f'Numerical (RK4 Shooting, h={h})', markersize=5)
plt.plot(t_fine, x_exact, 'k-', label='Exact Solution', linewidth=1.5)

plt.title("Boundary Value Problem 4.1 Solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.grid(True, alpha=0.3)
plt.legend()

print(f"数值解终点 x(3) = {x_num[-1]:.6f} (目标值: -1.0)")
plt.show()
