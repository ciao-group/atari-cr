""" Borrowed from https://github.com/
    ouyangzhibo/Image_Foveation_Python/blob/master/retina_transform.py """
from typing import Literal, TypeAlias
import cv2
import numpy as np
from numpy import ndarray

from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE


def gaussian_kernel(width: int, sigma: float):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel: ndarray = kernel / np.sum(kernel)
    return kernel

def pyramids(im: ndarray, sigma: float = 0.248, n: int = 6, upsample=True):
    """
    Generates Gaussian pyramids from an image.

    :param Array[W,H;u8] im: input image
    """
    start_height, start_width, ch = im.shape
    pyramid = im.copy()
    pyramids = [pyramid]

    # gaussian blur
    kernel = gaussian_kernel(5, sigma) # -> [5,5]

    # downsample
    # Half the side lengths with every iteration
    for i in range(1, n):
        pyramid = cv2.filter2D(pyramid, -1, kernel)
        if len(pyramid.shape) == 2: pyramid = pyramid[...,np.newaxis]
        height, width, _ = pyramid.shape
        pyramid = cv2.resize(pyramid, (int(width/2), int(height/2)))
        pyramids.append(pyramid)

    # upsample
    for i in range(1, n):
        pyramid = pyramids[i]
        for j in range(i):
            height, width = pyramid.shape[:2]
            im_size = (width * 2, height * 2)
            # Scale exactly to the original size for the last iteration
            if j == i - 1: im_size = (start_width, start_height)
            pyramid = cv2.resize(pyramid, im_size)
        pyramids[i] = pyramid if len(pyramid.shape) == 3 else pyramid[...,np.newaxis]
    pyramids = np.stack(pyramids)

    return pyramids

def foveat_img(im, fixs: list[tuple[int,int]]):
    """
    This function outputs the foveated image with given input image and fixations.

    :param Array[H,W,1;u8] im: input image
    :param list fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    """
    sigma=0.248
    n_pyramids = 6
    As = pyramids(im, sigma, n_pyramids)
    height, width, ch = im.shape

    k = 3
    # alpha: Half height angle, at 2.5° the visual resolution is halved
    alpha = 2.5

    # Visual degrees per pixel
    d = np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array([width, height])

    # Calculate the desired resolution R of each pixel
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    # Theta: Distances to the nearest fixation in °
    theta = np.full_like(x2d, np.inf)
    for fix in fixs:
        theta = np.minimum(
            theta, np.sqrt(
                ((x2d - fix[0]) * d[0]) ** 2
                + ((y2d - fix[1]) * d[1]) ** 2) )
    R = alpha / (theta + alpha)

    # Transfer function, maps R to the relative amplitude in the pyramids
    Ts = []
    for i in range(1, n_pyramids):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(R))

    # omega
    omega = np.zeros(n_pyramids)
    for i in range(1, n_pyramids):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, n_pyramids):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B: blending function, calculates  blendings coefficients for each pixel
    Bs = []
    for i in range(1, n_pyramids):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M, greyscale image indicating how much a pixel in each layer contributes to the
    # blurred image
    Ms = np.zeros((n_pyramids, R.shape[0], R.shape[1]))

    for i in range(n_pyramids):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    # generate periphery image
    im_fov = (As.transpose(3,0,1,2) * Ms).sum(axis=1).transpose(1,2,0)

    # Relative resolution of each pixel
    resolutions = np.stack(Ms / np.pow(4,np.arange(len(Ms)))[:,None,None]).sum(axis=0)

    im_fov = np.rint(im_fov).astype(np.uint8)
    return im_fov, resolutions

FovType: TypeAlias = Literal["window", "gaussian", "exponential", "window_periph"]

class Fovea():
    """
    :param bool weighting: Whether to also apply the relative resolution of a pixel
        as a weighting, putting more emphasis on high resolution pixels in neural
        networks
    """
    def __init__(self, type: FovType, size: tuple[int, int], weighting = False):
        self.type = type
        self.size = np.array(size, dtype=np.int32)
        self.weighting = weighting

    def apply(self, img_stack: ndarray, fixations: list[tuple[int, int]]):
        """ Applies the fovea to a stack of images in place

        :param Array[N,84,84;f64] img_stack: Stack of greyscale images
        """
        match self.type:
            case "window" | "window_periph":
                stack, w, h = img_stack.shape
                if self.type == "window_periph": # Fill area with the blurred orignal
                    assert w % 4 == 0 and h % 4 == 0, ("Only resolutions divisible by 4"
                        " are supported")
                    # Create periphery with 1/16 of the original information
                    masked_state = np.stack(
                        [Fovea.apply_periph(img) for img in img_stack])
                else: # Fill the area outside the crop with zeros
                    masked_state = np.full_like(img_stack, 0.5 if self.weighting else 0)

                assert self.size[0] % 2 == 0 and self.size[1] % 2 == 0, \
                    "Fov size has to be dividable by 2"
                for fixation in fixations:
                    (left, top), (right, bottom) = self.window(fixation, w, h)
                    crop = img_stack[..., top:bottom, left:right]
                    masked_state[..., top:bottom, left:right] = crop
                return masked_state

            case "gaussian":
                raise NotImplementedError()

            case "exponential":
                for i, img in enumerate(img_stack):
                    img = (img[...,None] * 255).astype(np.uint8)
                    img, resolutions = foveat_img(img.copy(), fixations)
                    img = (img[...,0]).astype(np.float64) / 255
                    if self.weighting:
                        # 0 weighted values are pushed towards 0.5
                        img = (img - 0.5) * resolutions + 0.5
                    img_stack[i] = img
                return img_stack

    def draw(self, frames: ndarray, fov_locs: ndarray, scaling = 1.):
        """ Draws the fovea onto a stack of frames

        :param list[Array[256,256,3]] frames:
        :param Array[N,2] fov_locs:
        :param float scaling: Scaling for the size of the fovea.
        """
        color = [0,255,0] # Green
        w, h, _ = frames[0].shape
        match(self.type):
            # Draw the window for a windowed fovea
            case "window" | "window_periph":
                for i in range(len(frames)):
                    top_left, bottom_right = self.window(fov_locs[i], w, h, scaling)
                    frames[i] = cv2.rectangle(
                        frames[i], top_left, bottom_right, color, 1)
            # Just mark the fov location for other foveae
            case "exponential":
                for i in range(len(frames)):
                    frames[i] = cv2.drawMarker(
                        frames[i], fov_locs[i].astype(np.int32), color)
            case _:
                raise NotImplementedError

    @staticmethod
    def _pixel_eccentricities(screen_size: tuple[int, int],
                              fixations: list[tuple[int, int]]):
        """
        Returns an array containing x and y distances from the fixations for every
        possible x and y coord. Used for broadcasted calculations.

        :param list fixations: List of N (x,y) coords
        :returns Array[N,2,W,H]:
        """
        N = len(fixations)
        x = np.arange(screen_size[0])
        y = np.arange(screen_size[1])
        mesh = np.stack(np.meshgrid(x, y, indexing="xy")) # -> [2,84,84]
        fixations = np.stack(fixations) # -> [N,2]
        fixations = np.broadcast_to(np.array(fixations)[
            ..., np.newaxis, np.newaxis], [N,2,*screen_size]) # -> [N,2,84,84]
        return mesh - fixations # -> [N,2,84,84]

    def window(self, fixation: tuple[int, int], width: int, height: int, scaling = 1.):
        """ Returns coords of the top left and bottom right window corners for the
        windowed fovea. """
        if self.type not in ["window", "window_periph"]:
            raise ValueError
        top_left = np.maximum(0, fixation - (self.size * scaling / 2)).astype(int)
        bottom_right = np.minimum(
            [width, height], fixation + (self.size * scaling / 2)).astype(int)
        return top_left, bottom_right

    @staticmethod
    def apply_periph(img):
        """
        Blurs an image by returning the thids Gaussian pyramid.

        :param Array[W,H;f32] img:
        :returns Array[W,H;f32]:
        """
        return pyramids(
            (img * 255).astype(np.uint8)[...,None]
        )[2,...,0].astype(float) / 255

    def get_visual_info(self, w: int, h: int, fixations: list[tuple[int,int]]) -> float:
        """ Returns a float between 0 and 1 for how much information from an image is
        retained on average, when the fovea is applied to different locations."""
        visual_infos = np.empty(len(fixations))
        for i,fix in enumerate(fixations):
            mask = Fovea("window", self.size).apply(np.ones((1,w,h)), fix)
            if self.type == "window":
                # Full information for the pixels inside the fovea
                visual_infos[i] = mask.mean()
            elif self.type == "window_periph":
                # Periphery with third pyramid level counts as 1/16 the information
                visual_infos[i] = mask.mean() + (1 - mask.mean()) / 16
            elif self.type == "exponential":
                _, resolutions = foveat_img(
                    mask.astype(np.uint8).transpose((1,2,0)), [fix])
                visual_infos[i] = resolutions.mean()
        return visual_infos.mean()

def EMMA_fixation_time(
        dist: float,
        freq = 0.1,
        execution_time = 0.070,
        K = 0.006,
        k = 0.4,
        saccade_scaling = 0.002,
        t_prep = 0.135,
    ):
    """
    Mathematical model for saccade duration in seconds from EMMA (Salvucci, 2001).
    Borrowed from https://github.com/aditya02acharya/TypingAgent/blob/master/src/utilities/utils.py.

    :param float dist: Eccentricity in visual degrees.
    :param float freq: Frequency of object being encoded. How often does the object
        appear. Value in (0,1).
    :param float execution_time: The base time it takes to execute an eye movement,
        independent of distance. 50ms for non-labile programming plus 20ms for saccade
        execution.
    :param float K: Scaling parameter for the encoding time.
    :param float k: Scaling parameter for the influence of the saccade distance on the
        encoding time.
    :param float saccade_scaling: Scaling parameter for the influence of the saccade
        distance on the execution time. 2ms per visual degree.
    :param float t_prep: Movement preparation time. If this is greater than the encoding
        time, no movement occurs. Was originally intended to be gamma distributed.

    :return EMMA_breakdown: tuple containing (preparation_time, execution_time,
        remaining_encoding_time).
    :return total_time: Total eye movement time in seconds.
    :return moved: true if encoding time > preparation time. false otherwise.
    """
    # visual encoding time
    t_enc = K * -np.log(freq) * np.exp(k * dist)

    # if encoding time < movement preparation time then no movement
    if t_enc < t_prep:
        return (t_enc, 0, 0), t_enc, False

    # movement execution time, originally intended to be gamma distributed
    t_exec = execution_time + saccade_scaling * dist
    # eye movement time (preparation time + execition time)
    t_sacc = t_prep + t_exec

    # if encoding time less then movement time
    if t_enc <= t_sacc:
        return (t_prep, t_exec, 0), t_sacc, True

    # if encoding left after movement time
    e_new = (k * -np.log(freq))
    t_enc_new = (1 - (t_sacc / t_enc)) * e_new

    return (t_prep, t_exec, t_enc_new), t_sacc + t_enc_new, True