import os
from atari_cr.atari_head.durations import get_histogram
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(0, 1025, 50)
    hist = get_histogram(game_name).numpy()

    # Only text annotation for the highest value
    highest = hist[1]
    hist[1] = 0
    second_highest = hist.max()
    plt.bar(x, hist, width=50, color="#74ADD1")
    plt.text(x[1], second_highest * 1.025, f"{highest:.3f}", ha='center', va='bottom')
    plt.ylim(0, second_highest * 1.1)

    # Put a single bar for the highest value
    y = np.zeros(21)
    y[1] = highest
    plt.bar(x, y, width=50, color="#FDAE61")

    # Styling
    plt.xticks(np.arange(0, 1050, 100))
    plt.title("Average distribution of frame durations per episode")
    plt.xlabel("Duration[ms]")

    plt.savefig(f"{output_dir}/{game_name}.png")
