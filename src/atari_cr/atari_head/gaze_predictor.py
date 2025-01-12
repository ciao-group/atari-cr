from collections import deque
import os
from typing import Callable, Optional
import cv2
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import h5py
from tap import Tap
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from atari_cr.atari_head.unet import UNet
from atari_cr.module_overrides import tqdm
from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.models import EvalResult

class ArgParser(Tap):
    debug: bool = False # Debug mode for less data loading
    load_model: bool = False # Whether to try and load an existing model
    n: int = 100 # Number of training iterations
    eval_train_data: bool = False # Whether to make an evaluation on the train data too
    load_saliency: bool = False # Whether to load existing saliency maps.
    unet: bool = False # Whether to use a unet instead of the conv deconv net
    unet_scale: int = 8 # Scaling factor for the size of the unet layers
    model_name: Optional[str] = None # Name of the model for saving
    dropout: float = 0.3 # Dropout rate for unet training
    ray: bool = False # Whether to use ray for multiple training runs

# @torch.compile()
class GazePredictionNetwork(nn.Module):
    """
    Conv deconv network predicting a saliency map for a given stack of
    4 greyscale atari game images.
    """
    def __init__(self, dropout = 0.25):
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
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Convolutional layers
        x = self.conv2d_1(x)
        x = self.batch_normalization_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2d_2(x)
        x = self.batch_normalization_2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2d_3(x)
        x = self.batch_normalization_3(x)
        x = self.act(x)
        x = self.dropout(x)

        # Deconvolutional layers
        x = self.conv2d_transpose_1(x)
        x = self.batch_normalization_4(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_2(x)
        x = self.batch_normalization_5(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_3(x)

        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.log_softmax(x)
        x = x.view(x.size(0), 84, 84)

        return x

    @staticmethod
    def from_h5(save_path: str, dropout=0.5):
        f = h5py.File(save_path, 'r')

        model = GazePredictionNetwork(dropout)
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

                value = torch.Tensor(value[:])
                value = value.permute(list(reversed(range(len(value.shape)))))
                key = key.replace("kernel", "weight")

                state_dict[f"{layer}.{key}"] = torch.Tensor(value)

        model.load_state_dict(state_dict)

        return model

class GazePredictor():
    """
    Wrapper around GazePredictionNetwork to handle training etc.

    :param torch.nn.Module model: Model predicting a [1,84,84] saliency map
        from a [4,84,84] stack of greyscale images
    """
    def __init__(self, model: nn.Module):
        self.model = model

        # Loss function, optimizer, compute device
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Count the number of trained epochs
        self.epoch = 0

    def train(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader,
              output_dir: str, save_interval: Optional[int] = None):
        if n_epochs == 0: return
        self.model.train()

        # Default args
        if not save_interval: save_interval = n_epochs // 10

        final_epoch = self.epoch + n_epochs
        losses = deque(maxlen=100)
        for self.epoch in range(self.epoch, final_epoch):

            print(f"Epoch {self.epoch + 1} / {final_epoch}")
            with tqdm(train_loader) as t:
                for inputs, targets in t:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    losses.append(loss.item())
                    t.set_postfix(loss=f"{np.mean(losses):6.3f}")

            # Regularly eval and save the model
            if self.epoch == final_epoch - 1 \
                or self.epoch % save_interval == save_interval - 1:

                if self.epoch == final_epoch - 1:
                    eval_df = self.baseline_eval(val_loader, train_loader)
                else:
                    eval = self.ray_eval(val_loader, train_loader)
                    eval_df = self._eval_df(eval)
                self.save(output_dir, eval_df)
                print(eval_df)

        self.model.eval()
        print('Training finished')

    def eval(self, loader: DataLoader,
             external_model: Callable[[torch.Tensor], torch.Tensor] = None, gt=False):
        """
        :param bool on_train_data: Whether to eval the predictor on the train set.
            For debugging purposes.
        :param Callable external_model: Model creating a saliency map in log space
        :param bool gt: Whether to produce a baseline evaluation using ground truth data
        :returns Tuple[float, float]: KL Divergence and AUC
        """
        model = external_model or self.model

        with torch.no_grad():

            min_val, max_val = 0, 0
            sums = torch.zeros(len(loader))
            kl_divs = torch.zeros(len(loader))
            aucs = torch.zeros(len(loader))
            entropies = torch.zeros(len(loader))
            for i, (frame_stack_batch, saliency_batch) in enumerate(loader):
                saliency_batch = saliency_batch.to(self.device)
                pred = saliency_batch if gt else model(
                    frame_stack_batch.to(self.device))
                if not gt: pred = pred.to(self.device).exp()

                # Non log space metrics
                aucs[i] = self.saliency_auc(
                    saliency_batch, pred, self.device, True).mean().item()
                sums[i] = pred.sum() / pred.size(0)
                entropies[i] = norm_entropy(pred.view(pred.size(0), -1), dim=1).mean()
                min_val = min(min_val, pred.min().item())
                max_val = max(max_val, pred.max().item())

                if gt: # Avoid 0 values for KLDiv
                    pred += torch.finfo(pred.dtype).eps
                    pred *= pred.size(0) / pred.sum()

                # KLDiv calculation in log space
                pred = pred.log()
                kl_divs[i] = nn.KLDivLoss(reduction="batchmean")(
                    pred, saliency_batch).item()

        r: EvalResult = {
            "min": min_val,
            "max": max_val,
            "sum": sums.mean().item(),
            "kl_div": kl_divs.mean().item(),
            "auc": aucs.mean().item(),
            "entropy": entropies.mean().item()
        }

        return r

    def baseline_eval(self, val_loader: DataLoader,
                      train_loader: Optional[DataLoader] = None):
        """
        Evluates the model against some baselines and returns a result dataframe.
        """
        print("Evaluating...")
        with tqdm(total=4) as pbar:
            evals = {}
            evals["Ground Truth"] = self.eval(val_loader, gt=True)
            pbar.update(1)

            # Model evaluation
            evals.update(self.ray_eval(val_loader, train_loader))
            pbar.update(1)

            # Baseline evaluation
            def optical_flow(t: torch.Tensor):
                """
                Wrapper to call the optical flow method from open cv on a validation set
                batch of data.

                :param Tensor[B x frame_stack x H x W] t: Batch of stacked frames
                :return Tensor[B x H x W]:
                """
                array = (t * 255).type(torch.uint8).detach().cpu().numpy()
                batch_size, _, height, width = t.shape
                saliency_batch = np.zeros([batch_size, height, width])
                for i, frame_stack in enumerate(array):
                    flow: np.ndarray = cv2.calcOpticalFlowFarneback(
                        frame_stack[-2], frame_stack[-1],None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # Interpret the absolute flow velocity as movement_saliency
                    flow = np.square(flow)
                    movement_saliency = np.sqrt(flow[..., 0] + flow[..., 1])
                    # Normalize the saliency map to sum to 1
                    movement_saliency = movement_saliency - movement_saliency.min()
                    movement_saliency = movement_saliency / movement_saliency.sum()
                    saliency_batch[i] = movement_saliency
                flow_saliency = torch.Tensor(saliency_batch).to(self.device)
                return nn.LogSoftmax(dim=1)(flow_saliency.view([batch_size, -1])).view(
                    [batch_size, width, height])
            evals["Optical Flow"] = self.eval(val_loader, external_model=optical_flow)
            pbar.update(1)

            def random_pred(t): return F.log_softmax(
                torch.rand([t.size(0),84*84]),dim=1).view([t.size(0),84,84])
            evals["Random"] = self.eval(val_loader, external_model=random_pred)
            pbar.update(1)

        return self._eval_df(evals)

    def ray_eval(self, val_loader: DataLoader,
                 train_loader: Optional[DataLoader] = None):
        """
        Evaluate the model on train and validation data and log the result to ray

        :returns dict[str, EvalResult]: Dict matching model name to EvalResult
        """
        val_eval = self.eval(val_loader)
        report = {
            "Eval KLDiv": val_eval["kl_div"],
            "Eval AUC": val_eval["auc"],
        }
        if train_loader:
            train_eval = self.eval(train_loader)
            report.update({
                "Train KLDiv": train_eval["kl_div"],
                "Train AUC": train_eval["auc"],
            })
        else: train_eval = None
        train.report(report)
        return { "Gaze Predictor (Train)": train_eval, "Gaze Predictor": val_eval }

    def save(self, output_dir: str, eval: Optional[pl.DataFrame] = None):
        """
        Saves the model checkpoint by its epoch as name

        :param dict eval: Eval data to save as yml next to the checkpoint
        """
        # Create output dir and paths
        checkpoint_dir = os.path.join(output_dir, str(self.epoch + 1))
        os.makedirs(checkpoint_dir, exist_ok=True)
        weights_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        eval_path = os.path.join(checkpoint_dir, "eval.csv")

        # Save the checkpoint
        torch.save(
            self.model.state_dict(),
            weights_path
        )
        if eval is not None:
            eval.write_csv(eval_path)
        print(f"Saved model checkpoint to {checkpoint_dir}")

    @staticmethod
    def from_save_file(save_path: str, model_class=GazePredictionNetwork):
        model = model_class()
        model.load_state_dict(torch.load(save_path, weights_only=False))

        predictor = GazePredictor(model)
        predictor.epoch = int(save_path.split("/")[-2])

        return predictor

    @staticmethod
    def _eval_df(eval_results: dict[str, EvalResult]):
        """ Creates a Dataframe from a dict of EvalResults"""
        model_names = pl.Series("model", list(eval_results.keys())).to_frame()
        return pl.concat([
                model_names,
                pl.from_dicts(list(eval_results.values()))],
            how="horizontal")

    @staticmethod
    def saliency_auc(gt_saliency: torch.Tensor, pred_saliency: torch.Tensor,
                     device: torch.device, threshold_gt=False):
        """
        Predicts the AUC between two greyscale saliency maps by comparing their most
        salient pixels each as described in doi.org/10.3758/s13428-012-0226-9.

        :param Tensor[BxWxH] ground_truth_saliency: Ground truth saliency map
        :param Tensor[BxWxH] predicted_saliency: Predicted saliency map
        :param bool threshold_gt: Whether to also have a changing threshold
            for the ground truth

        :return Tensor[Bx4]: AUC for most salient 2%, 5%, 10% and 20% of pixels
        """
        assert gt_saliency.shape == pred_saliency.shape, \
            "Saliency maps need to have the same size"

        # Flatten both maps
        batch_size, x, y = gt_saliency.shape
        n_pixels = x*y
        gt_saliency = gt_saliency.to(device).flatten(start_dim=1)
        pred_saliency = pred_saliency.to(device).flatten(start_dim=1)

        def percentile_saliency(saliency_map: torch.Tensor, percentiles: torch.Tensor):
            """ Get the q most salient pixels in the map for every fraction q in
            percentiles

            :param Tensor[B,WxH] saliency_map:
            :param Tensor[5] percentiles: """
            # Double precision to handle very small differences in thresholds
            saliency_map, percentiles = saliency_map.double(), percentiles.double()
            thresholds = torch.quantile(saliency_map, percentiles, dim=1) # -> [4,B]
            thresholds = thresholds.view(len(percentiles), batch_size, 1).expand(
                -1, -1, n_pixels) # -> [4,B,WxH]
            return saliency_map.expand(
                len(percentiles), batch_size, n_pixels) >= thresholds

        # Get the predicted saliency maps containing only the most salient pixels
        # for every qth percentile
        pred_percentiles = torch.arange(0.05, 1., 0.05).to(device)
        pred_percentile_saliency_maps = percentile_saliency(
            pred_saliency, pred_percentiles)

        if threshold_gt:
            gt_percentiles = pred_percentiles
        else:
            # Get the q most salient ground_truth pixels
            # for q in [2%, 5%, 10%, 15%, 20%, 50%]
            gt_percentiles = torch.Tensor(
                [0.98, 0.95, 0.90, 0.85, 0.80, 0.50]).to(device)
        gt_percentile_saliency_maps = percentile_saliency(gt_saliency, gt_percentiles)
            # -> [4,B,WxH]

        # Broadcast for comparison
        gt_percentile_saliency_maps = gt_percentile_saliency_maps.unsqueeze(0)
        pred_percentile_saliency_maps = pred_percentile_saliency_maps.unsqueeze(0)
        if not threshold_gt:
            gt_percentile_saliency_maps = gt_percentile_saliency_maps.transpose(0,1)
            gt_percentile_saliency_maps = gt_percentile_saliency_maps.expand(
                -1, len(pred_percentiles), -1, -1)
            pred_percentile_saliency_maps = pred_percentile_saliency_maps.expand(len(
                gt_percentiles), -1, -1, -1)
        return (gt_percentile_saliency_maps == pred_percentile_saliency_maps).type(
            torch.float32).mean(dim=[1,3]).clone()

def entropy(t: torch.Tensor, dim: Optional[int] = None):
    """ :param Tensor t: """
    t = (t + 1e-9) / (1 + t.numel() * 1e-9) # Avoid zeros
    return -(t*t.log()).sum(dim=dim)

def norm_entropy(t: torch.Tensor, dim: Optional[int] = None):
    """ Normalized entropy over a given dimension or the entire tensor
    :param Tensor t: """
    new_numel = 1 if dim is None else t.numel() / t.size(dim)
    return entropy(t, dim) / \
        entropy(torch.full(t.shape, new_numel/t.numel()).to(t.device), dim)

def train_predictor(args: ArgParser, dataset: Optional[GazeDataset] = None):
    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)
    np.random.seed(42)
    if args.debug: torch.cuda.memory._record_memory_history(True)

    # Create dataset and data loader
    env_name = "ms_pacman"
    single_run = "52_RZ_2394668_Aug-10-14-52-42" if args.debug else ""
    model_name = args.model_name or single_run or "all_trials"
    dataset = dataset or GazeDataset.from_atari_head_files(
        root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=single_run,
        load_saliency=args.load_saliency)
    train_loader, val_loader = dataset.split()

    # Create the dir for saving the trained model and its evaluations
    output_dir = f"output/atari_head/{env_name}"
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Load an existing Atari HEAD model
    model_files = os.listdir(model_dir)
    net = (lambda: UNet(4,1, args.unet_scale, args.dropout)) \
        if args.unet else (lambda: GazePredictionNetwork(args.dropout))
    if (args.load_model) and len(model_files) > 0:
        latest_epoch = sorted([int(file) for file in model_files])[-1]
        save_path = os.path.join(model_dir, str(latest_epoch), "checkpoint.pth")
        print(f"Loading existing gaze predictor from {save_path}")
        gaze_predictor = GazePredictor.from_save_file(save_path, net)
    else:
        print("Creating new gaze model")
        model = net() if args.unet else GazePredictionNetwork.from_h5(
            f"data/h5_gaze_predictors/{env_name}.hdf5")
        gaze_predictor = GazePredictor(model)

    # Train the model
    gaze_predictor.train(args.n, train_loader, val_loader, model_dir)

def tune_predictor(config, dataset: GazeDataset):
    args = ArgParser().from_dict({
        "debug": False,
        "load_model": False,
        "n": 1000,
        "eval_train_data": True,
        "load_saliency": True,
        "unet": True,
        "unet_scale": config["unet_scale"],
        "model_name": f"unet_{config['unet_scale']}",
        "dropout": config["dropout"],
    })
    print(dataset)
    train_predictor(args, dataset)

if __name__ == "__main__":
    args = ArgParser().parse_args()
    if args.ray:
        concurrent_runs = 2
        num_samples = 1 * concurrent_runs if args.debug else 50

        # Load the same dataset for all runs
        env_name = "ms_pacman"
        single_run = "52_RZ_2394668_Aug-10-14-52-42" if True else ""
        dataset = GazeDataset.from_atari_head_files(
            root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=single_run,
            load_saliency=True)

        trainable = tune_predictor
        trainable = tune.with_parameters(trainable, dataset=dataset)
        trainable = tune.with_resources( trainable,
                {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

        param_space = {
            "unet_scale": tune.grid_search([2]),
            "dropout": tune.grid_search([0.2])
        }

        metric, mode = ("Eval AUC", "max")
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=None if args.debug else ASHAScheduler(
                    stop_last_trials=True
                ),
                search_alg=OptunaSearch(),
                metric=metric,
                mode=mode
            ),
            run_config=train.RunConfig(
                storage_path="/home/niko/Repos/atari-cr/output/ray_results",
            )
        )
        results = tuner.fit()
        print("Best result:\n", results.get_best_result().config)
    else:
        train_predictor(args)
