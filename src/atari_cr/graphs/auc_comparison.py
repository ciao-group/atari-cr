import torch
from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.atari_head.gaze_predictor import GazePredictionNetwork, GazePredictor
from atari_cr.utils import debug_array

def compare_data_to_predictors(model801: GazePredictionNetwork):
    """ Compares saliency from the dataset to gaze predictor of 0.8 and 0.7 AUC """
    # Load 10 samples from the dataset
    data = GazeDataset.from_atari_head_files("data/Atari-HEAD/ms_pacman",
        # "52_RZ_2394668_Aug-10-14-52-42", load_saliency=True)
        "61_RZ_2737165_Aug-14-14-09-12", load_saliency=True)
    sample = next(iter(data.split(10)[1]))
    frame_stacks, saliency = [x.to("cuda") for x in sample]

    # Get gaze predictor saliency at .801 auc
    saliency801 = model801(frame_stacks).exp()

    # Get gaze predictor saliency at .725 auc
    saliency725 = GazePredictor.from_save_file(
        "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.5/1/checkpoint.pth"
    ).model(frame_stacks).exp()

    debug_array(torch.stack([saliency, saliency801, saliency725]),
                "output/graphs/gt-801-725.png")

def compare_predictor_to_agent(model801: GazePredictionNetwork):
    """ Compares saliency from the gaze predictor of 0.8 AUC to the agent """
    # TODO: Get saliency of an agent with 0.8 auc, or otherwise best agent
    frame_stacks, agent_saliency = \
        next(iter(GazeDataset.from_game_data([
                ""
            ]).to_loader(10)))

    # Get gaze predictor saliency at at .801 auc
    model_saliency = model801(frame_stacks).exp()

    debug_array(torch.stack([model_saliency, agent_saliency]),
                "output/graphs/801-agent.png")

if __name__ == "__main__":
    # Get gaze predictor at .801 auc
    model801 = GazePredictor.from_save_file(
        "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.3/999/checkpoint.pth"
    ).model

    compare_data_to_predictors(model801)
    # compare_predictor_to_agent(model801)
