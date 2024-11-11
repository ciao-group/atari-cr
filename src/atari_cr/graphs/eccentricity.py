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

# Found parameters of a simple exponential function through
# fovea, macula and periphery points
A, B, C = 2.0146, -0.4072, 0.5854
FIND_PARAMS = False

if __name__ == "__main__":
    if FIND_PARAMS:
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
    else:
        a, b, c = A, B, C

    # Plotting data
    x = np.arange(-80, 80, 0.1)
    y = func(np.abs(x), 0, a, b, c)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y)
    ax.grid()
    ticks = [np.arange(-80, 84, 4), np.arange(0, 3.2, 0.2)]
    tick_labels = [None, None]
    tick_labels[0] = [str(x) if x % 20 == 0 else "" for x in ticks[0]]
    tick_labels[1] = [str(int(x)) if x % 1 == 0 else "" for x in ticks[1]]
    ax.set_xticks(ticks[0], labels=tick_labels[0])
    ax.set_yticks(ticks[1], labels=tick_labels[1])
    ax.set_xlim([-80, 80])
    fig.subplots_adjust(top=0.96, bottom=0.16, left=0.06, right=0.98)
    ax.set_xlabel("Eccentricity (°)", fontweight="bold", fontsize="large")
    ax.set_ylabel("Visual acuity (decimal)", fontweight="bold", fontsize="large")
    out_path = "output/graphs/approx-eccentricity.png"
    fig.savefig(out_path)
    print(f"Plot saved under {out_path}")
