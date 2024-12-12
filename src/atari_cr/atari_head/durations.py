import os
import torch
import polars as pl

# Bins for values between [-inf, 25], [25, 75], ..., [925, 975], [975, inf]
BINS = torch.cat([
    torch.Tensor([-torch.inf]),
    torch.arange(25, 1000, 50, dtype=torch.float32),
    torch.Tensor([torch.inf])])

def get_durations(game: str):
    """ Get all gaze durations in one Atari-HEAD game.

    :param str game: Name of the atari game
    :return Tensor[21]:
    """
    game_dir = f"/home/niko/Repos/atari-cr/data/Atari-HEAD/{game}"
    if not os.path.exists(game_dir):
        raise IOError(f"Game directory {game_dir} does not exist")

    all_durations = []
    for csv_path in [f.path for f in os.scandir(game_dir) if f.name.endswith(".csv")]:
        durations = (
            pl.scan_csv(csv_path, null_values=["null"])
            .select(pl.col("duration(ms)"))
            .drop_nulls()
            .collect()
            .to_series()
            .to_numpy()
        )
        all_durations.append(torch.Tensor(durations))
    return torch.concat(all_durations)

def get_histogram(game: str):
    """ Get a histogram for the average episode gaze durations per timestep in one
    Atari-HEAD game.

    :param str game: Name of the atari game
    :return Tensor[21]:
    """
    durations = get_durations(game)
    return torch.histogram(durations, BINS).hist / len(durations)


if __name__ == "__main__":
    h = get_histogram("ms_pacman")

    from matplotlib import pyplot as plt
    plt.bar(BINS[1:].numpy() - 25, h.numpy(), width=50)
    plt.xticks(torch.arange(0, 1001, 100).numpy())
    plt.savefig("debug.png")
