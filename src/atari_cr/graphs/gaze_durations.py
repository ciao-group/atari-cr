import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
from atari_cr.graphs.colors import ColorBlind

AGENT_DURATIONS = np.array([
    13.815511, 1085.2463, 1186.3843, 2882.3455, 1196.609, 1139.8745,
    1005.7478,1025.7812,1155.9268,1131.9165,2257.9004,1148.9196,
    1011.96576, 1144.0747,1856.3909,3630.389,  218.78574, 1148.9196,
    1176.6908,1092.846, 1057.5708,1906.2853,1005.7478,1091.1405,
    1178.9528, 724.47437, 1350.7023,  40.915386, 724.47437, 1155.0875,
    75.55339, 3423.2688,1187.6119,2231.2776, 766.41077, 1092.846,
    2167.0532,  40.915386, 2091.1472,1957.0205,1119.2618,1161.2206,
    103.76305, 3063.4783,1163.1914,1057.5708, 218.78574, 75.55339,
    1921.4983, 880.0913,1166.7599, 574.3791,1142.2025,1011.96576,
    1011.96576, 1005.7478,2236.9187,1163.6747,1166.7599,1176.3676,
    1176.6908,1142.2025,  75.55339, 1148.9196,  75.55339, 1386.0807,
    1071.3864,1162.735,  870.1738,  40.915386,75.55339, 1183.613,
    1169.6572,1011.96576,121.17315, 1057.5708,1155.0875,2252.361,
    1166.7599,1177.6392,1152.3868,1148.9196,3016.5952,2400.4087,
    1161.2206,1139.8745,1092.846, 2334.9248,1166.7599, 880.0913,
    103.76305, 2346.874, 1119.2618, 103.76305, 1139.8745,1224.5336,
    2225.1208,1109.0016, 880.0913, 648.1796,1140.7312,1165.1373,
    1183.613, 1185.3801,  13.815511, 1176.6908,  13.815511,75.55339,
    880.0913,  40.915386, 1142.2025,1186.3843,1257.6069,1005.7478,
    3320.9072,2284.806, 1005.7478,1171.8492,1005.7478, 437.57147,
    1131.9165,4573.2725, 218.78574,800.0278,1809.7207,1092.846,
    218.78574, 40.915386,13.815511, 1119.2618, 951.27264, 2305.394,
    218.78574, 2342.3945, 218.78574, 1166.7599, 951.27264,766.41077,
    1068.0862,  13.815511, 1139.8745,1191.3645,1135.1316, 103.76305,
    1005.7478,1165.7036, 880.0913,1119.2618, 724.47437, 1139.8745,
    1119.2618,1068.0862,  13.815511, 1131.9165,2019.3589, 880.0913,
    75.55339, 1092.846,  965.0882,1057.5708,2069.5366,2164.3525,
    1101.7872,1166.7599,1140.7312,1161.2206, 724.47437, 2130.7373,
    1092.846, 1170.0151, 648.1796  ])

if __name__ == "__main__":
    agent = True
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(0, 1025, 50)
    hist = np.histogram(AGENT_DURATIONS, BINS)[0] if agent \
        else get_histogram(game_name).numpy()
    hist: np.ndarray = hist / hist.sum()

    # Only text annotation for the highest value
    highest = hist.max()
    highest_idx = hist.argmax()
    hist[highest_idx] = 0
    second_highest = hist.max()
    plt.text(x[highest_idx], second_highest * 1.025, f"{highest:.3f}", ha='center',
            va='bottom')
    plt.ylim(0, second_highest * 1.1)

    # Put a single bar for the highest value
    y = np.zeros(21)
    y[highest_idx] = highest
    plt.bar(x, y, width=50, color=ColorBlind.ORANGE)

    # Normal plot
    plt.bar(x, hist, width=50, color=ColorBlind.BLUE)

    # Styling
    plt.xticks(np.arange(0, 1050, 100))
    plt.title(("Agent" if agent else "Human") + " frame durations per episode")
    plt.xlabel("Duration[ms]")

    plt.savefig(f"{output_dir}/{game_name}_agent.png" if agent \
        else f"{output_dir}/{game_name}.png")
