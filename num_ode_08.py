import numpy as np
import matplotlib.pyplot as plt


def p_func(t): return -4.0


def q_func(t): return -4.0


def r_func(t): return 5.0 * np.cos(4.0 * t) + np.sin(2.0 * t)


def thomas_tridiagonal(a, b, c, d):
    """托马斯算法求解三对角矩阵系统"""
    n = len(d)
    ac, bc, cc, dc = a.astype(float).copy(), b.astype(float).copy(), c.astype(float).copy(), d.astype(float).copy()
    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def solve_bvp_fd(h, t_range=(0.0, 2.0), x_bounds=(0.75, 0.25)):
    """有限差分法核心程序"""
    a_limit, b_limit = t_range
    alpha, beta = x_bounds
    N = int(round((b_limit - a_limit) / h))
    t = np.linspace(a_limit, b_limit, N + 1)

    n_unknown = N - 1
    a_diag, b_diag, c_diag, d_rhs = np.zeros(n_unknown), np.zeros(n_unknown), np.zeros(n_unknown), np.zeros(n_unknown)

    for j in range(1, N):
        tj = t[j]
        pj, qj, rj = p_func(tj), q_func(tj), r_func(tj)

        idx = j - 1
        a_diag[idx] = -0.5 * h * pj - 1.0
        b_diag[idx] = 2.0 + (h ** 2) * qj
        c_diag[idx] = 0.5 * h * pj - 1.0
        d_rhs[idx] = -(h ** 2) * rj

    d_rhs[0] -= a_diag[0] * alpha
    a_diag[0] = 0.0
    d_rhs[-1] -= c_diag[-1] * beta
    c_diag[-1] = 0.0

    x_interior = thomas_tridiagonal(a_diag, b_diag, c_diag, d_rhs)
    x = np.concatenate(([alpha], x_interior, [beta]))
    return t, x


def exact_solution(t):
    """题目给出的精确解公式"""
    term1 = -1 / 40 + 1.025 * np.exp(-2 * t) - 1.915729975 * t * np.exp(-2 * t)
    term2 = 19 / 20 * np.cos(t) ** 2 - 6 / 5 * np.cos(t) ** 4 - 4 / 5 * np.cos(t) * np.sin(t) + 8 / 5 * np.cos(
        t) ** 3 * np.sin(t)
    return term1 + term2


h = 0.05
t_num, x_num = solve_bvp_fd(h)
t_fine = np.linspace(0, 2, 500)
x_exact = exact_solution(t_fine)

plt.figure(figsize=(10, 6))
plt.plot(t_num, x_num, 'ro', label=f'Numerical (Finite Difference, h={h})', markersize=4)
plt.plot(t_fine, x_exact, 'k-', label='Exact Solution', linewidth=1.5)
plt.title("Boundary Value Problem 4.3 Solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"数值解边界确认: x(0)={x_num[0]:.4f}, x(2)={x_num[-1]:.4f}")
