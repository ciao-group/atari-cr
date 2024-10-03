from collections import deque
import os
from typing import Callable, Optional
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import h5py
from tap import Tap
from atari_cr.common.module_overrides import tqdm

from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.common.utils import gradfilter_ema, debug_array

class ArgParser(Tap):
    debug: bool = False # Debug mode for less data loading
    load_model: bool = False # Whether to load an existing model (if possible) or train a new one
    n: int = 100 # Number of training iterations
    eval_train_data: bool = False # Whether to use the train data for evaluation as well. For debugging
    eval_tf: bool = False # Whether to also evaluate the original tensorflow model
    load_saliency: bool = False # Whether to load existing saliency maps.
    use_og_saliency: bool = False # Whether to use the original funtion for creating saliency maps

class GazePredictionNetwork(nn.Module):
    """
    Neural network predicting a saliency map for a given stack of 4 greyscale atari game images.
    """
    def __init__(self):
        super(GazePredictionNetwork, self).__init__()
        
        # Convolutional layers
        self.conv2d_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.batch_normalization_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batch_normalization_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batch_normalization_3 = nn.BatchNorm2d(64)
        
        # Deconvolutional (transpose convolution) layers
        self.conv2d_transpose_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.batch_normalization_4 = nn.BatchNorm2d(64)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.batch_normalization_5 = nn.BatchNorm2d(32)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4)
        
        # Softmax layer; Uses log softmax to conform to the KLDiv expected input
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        # Convolutional layers
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.batch_normalization_1(x)
        x = self.dropout(x)
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = self.batch_normalization_2(x)
        x = self.dropout(x)
        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.batch_normalization_3(x)
        x = self.dropout(x)
        
        # Deconvolutional layers
        x = self.conv2d_transpose_1(x)
        x = F.relu(x)
        x = self.batch_normalization_4(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_2(x)
        x = F.relu(x)
        x = self.batch_normalization_5(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_3(x)
        
        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.log_softmax(x)
        x = x.view(x.size(0), 84, 84)
        
        return x
    
    @staticmethod
    def from_h5(save_path: str):
        f = h5py.File(save_path, 'r')

        model = GazePredictionNetwork()
        state_dict = model.state_dict()

        h5_weights = {}
        for key in f["model_weights"]:
            if len(f["model_weights"][key]) > 0:
                h5_weights[key] = f["model_weights"][key][key]

        for layer in h5_weights:
            for key in h5_weights[layer]:
                value = h5_weights[layer][key]
                key: str = key[:-2]
                key = key.replace("gamma", "weight") \
                    .replace("beta", "bias") \
                    .replace("moving", "running") \
                    .replace("variance", "var") \
                    .replace("kernel", "weight")

                if not isinstance(value[:], np.ndarray): breakpoint()
                value = torch.Tensor(value[:])
                value = value.permute(list(reversed(range(len(value.shape)))))
                key = key.replace("kernel", "weight")

                state_dict[f"{layer}.{key}"] = torch.Tensor(value)

        # OPTIONAL: Optimizer weights

        model.load_state_dict(state_dict)

        return model
    
class GazePredictor():
    """
    Wrapper around GazePredictionNetwork to handle training etc.

    :param torch.nn.Module model: Model predicting a [1,84,84] saliency map from a [4,84,84] stack 
        of greyscale images
    """
    def __init__(
            self, 
            model: nn.Module,
            dataset: GazeDataset,
            output_dir: str,
            model_name: str = "all_trials"
        ):
        self.model = model
        self.train_loader, self.val_loader = dataset.split(batch_size=512)
        self.max_gazes = dataset.max_gazes
        self.output_dir = output_dir
        self.model_name = model_name

        # Loss function, optimizer, compute device and tesorboard writer
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

        # Init Grokfast
        self.grokfast = False
        self.grads = None

        # Count the number of trained epochs
        self.epoch = 0

    def train(self, n_epochs: int, save_interval=100):
        if n_epochs == 0: return
        self.model.train()

        final_epoch = self.epoch + n_epochs
        losses = deque(maxlen=100)
        eval_kl_divs, train_kl_divs = [], []
        for self.epoch in range(self.epoch, final_epoch):

            print(f"Epoch {self.epoch + 1} / {final_epoch}")
            with tqdm(self.train_loader) as t:
                for inputs, targets in t:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    loss.backward()
                    if self.grokfast: self.grads = gradfilter_ema(self.model, self.grads)
                    self.optimizer.step()

                    losses.append(loss.item())
                    t.set_postfix(loss=f"{np.mean(losses):6.4f}")

            if self.epoch % save_interval == save_interval - 1:
                self.save()

                kl_div, auc = self.eval()
                eval_kl_divs.append(kl_div.item())
                print(f"Eval KLDivs: {[f'{x:5.3f}' for x in eval_kl_divs]}")
                print(f"Eval AUC: {auc}")
                kl_div, auc = self.eval(on_train_data=True)
                train_kl_divs.append(kl_div.item())
                print(f"Train KLDivs: {[f'{x:5.3f}' for x in train_kl_divs]}")
                print(f"Tain AUC: {auc}")

        self.model.eval()
        print('Training finished')
        
        # Save the trained model
        self.save()

    def eval(self, on_train_data=False, external_model: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        :param bool on_train_data: Whether to eval the predictor on the train set. For debugging purposes.
        :param Callable external_model: Model creating a saliency map in log space
        :returns Tuple[float, float]: KL Divergence and AUC
        """
        loader = self.train_loader if on_train_data else self.val_loader
        model = external_model or self.model
        if on_train_data: print("Warning: Evaluating on training data")
        kl_divs = torch.zeros(len(loader))
        aucs = [None] * len(loader)
        for i, (frame_stack_batch, saliency_map_batch) in enumerate(loader):
            # Prediction in log space as expected by KLDivLoss
            prediction = model(frame_stack_batch.to(self.device))
            saliency_map_batch = saliency_map_batch.to(self.device)

            kl_divs[i] = nn.KLDivLoss(reduction="batchmean")(prediction, saliency_map_batch).detach()
            aucs[i] = self.saliency_auc(saliency_map_batch, prediction.exp(), self.device).mean(dim=1)

        return kl_divs.mean(), torch.stack(aucs).mean(dim=0)
    
    def save(self):
        save_path = f"{self.epoch + 1}.pth"
        save_dir = os.path.join(self.output_dir, "models", self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(), 
            os.path.join(save_dir, save_path)
        )
        print(f"Saved model to {save_path}")
    
    @staticmethod
    def from_save_file(save_path: str, dataset: GazeDataset, output_dir: str, model_class=GazePredictionNetwork):
        model = model_class()
        model.load_state_dict(torch.load(save_path, weights_only=False))

        model_name = save_path.split("/")[-2]
        predictor = GazePredictor(model, dataset, output_dir, model_name)
        predictor.epoch = int(save_path.split("/")[-1][:-4])

        return predictor
    
    @staticmethod
    def saliency_auc(gt_saliency: torch.Tensor, pred_saliency: torch.Tensor, device: torch.device, threshold_gt=False):
        """
        Predicts the AUC between two greyscale saliency maps by comparing their most salient pixels each
        as described in doi.org/10.3758/s13428-012-0226-9.

        :param Tensor[BxWxH] ground_truth_saliency: Ground truth saliency map
        :param Tensor[BxWxH] predicted_saliency: Predicted saliency map
        :param bool threshold_gt: Whether to also have a changing threshold for the ground truth

        :return Tensor[Bx4]: AUC for most salient 2%, 5%, 10% and 20% of pixels
        """
        assert gt_saliency.shape == pred_saliency.shape, "Saliency maps need to have the same size"

        # Flatten both maps
        batch_size, x, y = gt_saliency.shape
        n_pixels = x*y
        gt_saliency = gt_saliency.to(device).flatten(start_dim=1)
        pred_saliency = pred_saliency.to(device).flatten(start_dim=1)

        def percentile_saliency(saliency_map: torch.Tensor, percentiles: torch.Tensor):
            """ Get the q most salient pixels in the map for every fraction q in percentiles 

            :param Tensor[B,WxH] saliency_map: 
            :param Tensor[5] percentiles: """
            # Double precision to handle very small differences in thresholds
            saliency_map, percentiles = saliency_map.double(), percentiles.double()
            thresholds = torch.quantile(saliency_map, percentiles, dim=1) # -> [4,B]
            thresholds = thresholds.view(len(percentiles), batch_size, 1).expand(-1, -1, n_pixels) # -> [4,B,WxH]
            return saliency_map.expand(len(percentiles), batch_size, n_pixels) >= thresholds

        # Get the predicted saliency maps containing only the most salient pixels for every qth percentile
        pred_percentiles = torch.arange(0.05, 1., 0.05).to(device)
        pred_percentile_saliency_maps = percentile_saliency(pred_saliency, pred_percentiles)
        # debug_array(pred_percentile_saliency_maps.view(-1,16,84,84))

        if threshold_gt:
            gt_percentiles = pred_percentiles
        else:
            # Get the q most salient ground_truth pixels for q in [2%, 5%, 10%, 15%, 20%, 50%]
            gt_percentiles = torch.Tensor([0.98, 0.95, 0.90, 0.85, 0.80, 0.50]).to(device)
        gt_percentile_saliency_maps = percentile_saliency(gt_saliency, gt_percentiles) # -> [4,B,WxH]
        # debug_array(gt_percentile_saliency_maps.view(-1,16,84,84))

        # Broadcast for comparison
        gt_percentile_saliency_maps = gt_percentile_saliency_maps.unsqueeze(0)
        pred_percentile_saliency_maps = pred_percentile_saliency_maps.unsqueeze(0)
        if not threshold_gt:
            gt_percentile_saliency_maps = gt_percentile_saliency_maps.transpose(0,1)
            gt_percentile_saliency_maps = gt_percentile_saliency_maps.expand(-1, len(pred_percentiles), -1, -1)
            pred_percentile_saliency_maps = pred_percentile_saliency_maps.expand(len(gt_percentiles), -1, -1, -1)

        return (gt_percentile_saliency_maps == pred_percentile_saliency_maps).type(torch.float32).mean(dim=[1,3])
    
def entropy(t: torch.Tensor, dim: Optional[int] = None):
    """ :param Tensor t: """
    t = (t + 1e-9) / (1 + t.numel() * 1e-9) # Avoid zeros
    return -(t*t.log()).sum(dim=dim)

def norm_entropy(t: torch.Tensor, dim: Optional[int] = None):
    """ Normalized entropy for a batch of distributions 
    :param Tensor[B,*] t: """
    new_numel = 1 if dim is None else t.numel() / t.size(dim)
    return entropy(t, dim) / entropy(torch.full(t.shape, new_numel/t.numel()).to(t.device), dim)

def train_predictor():
    args = ArgParser().parse_args()

    # Use bfloat16 to speed up matrix computation
    # torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)
    np.random.seed(42)
    if args.debug: torch.cuda.memory._record_memory_history(True)

    # Create dataset and data loader
    env_name = "ms_pacman"
    single_run = "52_RZ_2394668_Aug-10-14-52-42" if args.debug else ""
    model_name = single_run or "all_trials"
    dataset = GazeDataset.from_atari_head_files(
        root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=single_run, 
        load_saliency=args.load_saliency, use_og_saliency=args.use_og_saliency)
    
    # Create the dir for saving the trained model
    output_dir = f"output/atari_head/{env_name}"
    model_dir = os.path.join(output_dir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Load an existing Atari HEAD model
    model_files = os.listdir(model_dir)
    if (args.load_model) and len(model_files) > 0:
        latest_epoch = sorted([int(file[:-4]) for file in model_files])[-1]
        save_path = os.path.join(model_dir, f"{latest_epoch}.pth")
        print(f"Loading existing gaze predictor from {save_path}")
        gaze_predictor = GazePredictor.from_save_file(save_path, dataset, output_dir)
    else:
        print("Creating new gaze model from hdfs5 weights")
        model = GazePredictionNetwork.from_h5(f"data/h5_gaze_predictors/{env_name}.hdf5")
        gaze_predictor = GazePredictor(model, dataset, output_dir, model_name)

    # Train the model
    gaze_predictor.train(n_epochs=args.n)
    # model_kl_div, model_auc = gaze_predictor.eval()
    # print(f"KL Divergence: {model_kl_div:6.4f}, AUC: {model_auc}")
    if args.eval_train_data: 
        train_kl_div, train_auc = gaze_predictor.eval(args.eval_train_data)
        print(f"Evaluation on Training Data: KL Divergence: {train_kl_div:6.4f}, AUC: {train_auc}")

    # Baseline evaluation
    def optical_flow(t: torch.Tensor):
        """
        Wrapper to call the optical flow method from open cv on a validation set batch of data.

        :param Tensor[B x frame_stack x H x W] t: Batch of stacked frames
        :return Tensor[B x H x W]:
        """
        array = (t * 255).type(torch.uint8).detach().cpu().numpy()
        batch_size, _, height, width = t.shape
        saliency_batch = np.zeros([batch_size, height, width])
        for i, frame_stack in enumerate(array):
            flow: np.ndarray = cv2.calcOpticalFlowFarneback(frame_stack[-2], frame_stack[-1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # Interpret the absolute flow velocity as movement_saliency
            flow = np.square(flow)
            movement_saliency = np.sqrt(flow[..., 0] + flow[..., 1])
            # Normalize the saliency map to sum to 1
            movement_saliency = movement_saliency - movement_saliency.min()
            movement_saliency = movement_saliency / movement_saliency.sum()
            saliency_batch[i] = movement_saliency
        flow_saliency = torch.Tensor(saliency_batch).to(gaze_predictor.device)
        return nn.LogSoftmax(dim=1)(flow_saliency.view([batch_size, -1])).view([batch_size, width, height])
    # flow_kl_div, flow_auc = gaze_predictor.eval(external_model=optical_flow)
    # print(f"Baseline Evaluation: KL Divergence: {flow_kl_div:6.4f}, AUC: {flow_auc:6.4f}")

    if args.debug:
        # Get one validation batch for debugging
        frame_stack, gt_saliency = next(iter(gaze_predictor.val_loader))
        frame_stack = frame_stack[16:32].to(gaze_predictor.device)
        gt_saliency = gt_saliency[16:32].to(gaze_predictor.device)
        # Add small number to 0 values for KLDiv to work and make it add to one per batch again
        gt_saliency += torch.finfo(gt_saliency.dtype).eps 
        gt_saliency *= gt_saliency.size(0) / gt_saliency.sum()

        # Look at output of different models
        model_saliency = gaze_predictor.model(frame_stack).exp().detach()
        baseline_saliency = optical_flow(frame_stack).exp()
        random_saliency = torch.rand(gt_saliency.shape).to(gaze_predictor.device)
        random_saliency = nn.Softmax(dim=1)(random_saliency.view(gt_saliency.shape[0], -1)).view(gt_saliency.shape)
        models = {"Saliency": [gt_saliency, model_saliency, baseline_saliency, random_saliency]}

        # KLDiv
        kldiv_loss = lambda t: nn.KLDivLoss(reduction="batchmean")(t.log(), gt_saliency).item()
        models["KLDiv"] = list(map(kldiv_loss, models["Saliency"]))
        # AUC, [2%, 5%, 10%, 15%, 20%, 50%]
        auc_score = lambda t: GazePredictor.saliency_auc(gt_saliency, t, gaze_predictor.device, True).mean(dim=1)[0].item()
        models["AUC"] = list(map(auc_score, models["Saliency"]))
        # Normalized Entropy
        n_entropy = lambda t: norm_entropy(t.view(t.size(0), -1), dim=1).mean().item()
        models["Normalized Entropy"] = list(map(n_entropy, models["Saliency"]))

        model_df = pd.DataFrame({
            "Min": list(map(lambda t: t.min().item(), models["Saliency"])),
            "Max": list(map(lambda t: t.max().item(), models["Saliency"])),
            "Sum": list(map(lambda t: t.sum().item(), models["Saliency"])),
            "KLDiv": models["KLDiv"],
            "T AUC": models["AUC"],
            "Norm. Entropy": models["Normalized Entropy"]
        }, index=["Ground Truth", "Gaze Predictor", "Optical Flow", "Random"])
        print(model_df)
 
        debug_array(models["Saliency"])  
        print(f"Image has been saved to 'debug.png' with rows: GT,Model,Flow")

    # Eval original tensorflow model
    if args.eval_tf:
        with open("output/debug/ms_pacman_52_RZ_2394668_Aug-10-14-52-42_preds.np", "rb") as f: 
            tf_preds = np.load(f).copy()
        ground_truth_saliency = np.stack(list(dataset.data["saliency_map"]))

        debug_array(np.stack([tf_preds[:16], ground_truth_saliency[:16]]))    

if __name__ == "__main__": 
    train_predictor()
