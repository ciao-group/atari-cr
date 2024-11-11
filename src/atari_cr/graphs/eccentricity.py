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
    [3.3, 1.0],
    [5.5, 0.8],
    [12.1, 0.6],
    [80, 0.3]
])

def func(x, y, a, b, c, d, e):
    return a * np.exp(x * b) + c - y + np.log(x * d) * e

def system_of_equations(variables):
    a, b, c, d, e = variables
    return [func(p[0], p[1], a, b, c, d, e) for p in POINTS]

# Initial guess for the variables
initial_guess = [1.0, -1.0, 0.0, 1.0, -1.0]

# Solve the system of equations
a, b, c, d, e = fsolve(system_of_equations, initial_guess)

print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"c = {c:.4f}")
print(f"d = {d:.4f}")
print(f"e = {e:.4f}")

x = np.arange(-80, 80, 0.1)
y = func(np.abs(x), 0, a, b, c, d, e)
# y = a * np.exp(-b * np.abs(x)) + c + np.square(np.abs(x)) * d + np.abs(x) * e

plt.plot(x, y)
plt.savefig("debug.png")
print("Plot saved under debug.png")
