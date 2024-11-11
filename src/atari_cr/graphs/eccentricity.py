from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# fov: 0, 2.6
# parafov: 3.3, 1.0
# macula: 5.5, 0.8
# periphery: 12.1, 0.6
# global: 80, 0.3
POINTS = np.array([
    [0, 2.6],
    # [3.3, 1.0],
    [5.5, 0.8],
    [12.1, 0.6],
    # [80, 0.3]
])

def func(x, y, a, b, c):
    return a * np.exp(b * x) + c - y

def system_of_equations(variables):
    a, b, c = variables
    return [func(p[0], p[1], a, b, c) for p in POINTS]

# Initial guess for the variables
initial_guess = [1.8, -1.0, 0.8]

# Solve the system of equations
a, b, c = fsolve(system_of_equations, initial_guess)

print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"c = {c:.4f}")

x = np.arange(-80, 80, 0.1)
y = func(np.abs(x), 0, a, b, c)

plt.plot(x, y)
out_path = "output/graphs/eccentricity.png"
plt.savefig(out_path)
print(f"Plot saved under {out_path}")
