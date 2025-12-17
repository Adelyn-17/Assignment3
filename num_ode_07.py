import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def system_4_2(t, U):
    """
    U[0] = x
    U[1] = x'
    """
    u1, u2 = U
    du1dt = u2
    du2dt = -2.0 * u2 - 2.0 * u1 + np.exp(-t) + np.sin(2.0 * t)
    return np.array([du1dt, du2dt])


def rk4_solve(f, t_span, u0, h):
    t0, T = t_span
    N = int(np.round((T - t0) / h))
    t = np.linspace(t0, T, N + 1)
    U = np.zeros((N + 1, len(u0)))
    U[0] = u0

    for k in range(N):
        tk = t[k]
        Uk = U[k]
        k1 = f(tk, Uk)
        k2 = f(tk + 0.5 * h, Uk + 0.5 * h * k1)
        k3 = f(tk + 0.5 * h, Uk + 0.5 * h * k2)
        k4 = f(tk + h,       Uk + h * k3)
        U[k + 1] = Uk + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return t, U

t0, t_end = 0.0, 4.0
x_start = 0.6    
x_target = -0.1     
h = 0.05

def objective(s):
    u0 = [x_start, float(s)]
    ts, us = rk4_solve(system_4_2, [t0, t_end], u0, h)
    return us[-1, 0] - x_target

correct_s = fsolve(objective, 0.0)[0]
print(f"找到的初始斜率 x'(0) = {correct_s:.6f}")


def exact_solution(t):
    c_const = 3.670227413
    term1 = 1/5 + np.exp(-t)
    term2 = - (1/5) * np.exp(-t) * np.cos(t)
    term3 = - (2/5) * (np.cos(t)**2)
    term4 = c_const * np.exp(-t) * np.sin(t)
    term5 = - (1/5) * np.cos(t) * np.sin(t)
    return term1 + term2 + term3 + term4 + term5

t_num, u_num = rk4_solve(system_4_2, [t0, t_end], [x_start, correct_s], h)
x_num = u_num[:, 0]

t_fine = np.linspace(t0, t_end, 500)
x_exact_vals = exact_solution(t_fine)

plt.figure(figsize=(10, 6))
plt.plot(t_num, x_num, 'ro', label=f'Numerical (RK4 Shooting, h={h})', markersize=4)
plt.plot(t_fine, x_exact_vals, 'k-', label='Exact Solution', linewidth=1.5)

plt.title("BVP Problem 4.2 Solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.grid(True, alpha=0.3)
plt.legend()

print(f"数值解边界确认: x(0)={x_num[0]:.4f}, x(4)={x_num[-1]:.4f}")
plt.show()
