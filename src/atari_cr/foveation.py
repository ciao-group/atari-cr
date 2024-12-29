""" Borrowed from https://github.com/
    ouyangzhibo/Image_Foveation_Python/blob/master/retina_transform.py """
from typing import Literal, TypeAlias
import cv2
import numpy as np

from atari_cr.atari_head.utils import VISUAL_DEGREES_PER_PIXEL
from atari_cr.graphs.eccentricity import A, B, C


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im: np.ndarray, sigma: float = 0.248, prNum: int = 6):
    """ :param Array[W,H;u8] im: input image """
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]

    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma) # -> [5,5]

    # downsample
    # Half the side lengths with every iteration
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        if len(G.shape) == 2: G = G[...,np.newaxis]
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)

    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im if len(curr_im.shape) == 3 else curr_im[...,np.newaxis]
    pyramids = np.stack(pyramids)

    return pyramids

def foveat_img(im, fixs: list[tuple[int,int]]):
    """
    This function outputs the foveated image with given input image and fixations.

    :param Array[W,H;u8] im: input image
    :param list fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    """
    sigma=0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, ch = im.shape

    k = 3
    # alpha: Half height angle, at 2.5° the visual resolution is halved
    alpha = 2.5

    # Calculate the desired resolution R of each pixel
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    # Theta: Distances to the nearest fixation in °
    theta = np.full_like(x2d, np.inf)
    for fix in fixs:
        theta = np.minimum(
            theta, np.sqrt(
                ((x2d - fix[0]) * VISUAL_DEGREES_PER_PIXEL[0]) ** 2
                + ((y2d - fix[1]) * VISUAL_DEGREES_PER_PIXEL[1]) ** 2) )
    R = alpha / (theta + alpha)

    # Transfer function, maps R to the relative amplitude in the pyramids
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(R))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B: blending function, calculates  blendings coefficients for each pixel
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M, greyscale image indicating how much a pixel in each layer contributes to the
    # blurred image
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
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

FovType: TypeAlias = Literal["window", "gaussian", "exponential"]

class Fovea():
    """
    :param bool weighting: Whether to also apply the relative resolution of a pixel
        as a weighting, putting more emphasis on high resolution pixels in neural
        networks
    """
    def __init__(self, type: FovType, size: tuple[int, int], periph = False,
                 weighting = False):
        self.type = type
        self.size = np.array(size, dtype=np.int32)
        self.periph = periph
        self.weighting = weighting

    def apply(self, img_stack: np.ndarray, fixations: list[tuple[int, int]]):
        """ Applies the fovea to a stack of images in place

        :param Array[4,84,84;f64] img_stack: Stack of four greyscale images
        """
        match self.type:
            case "window":
                stack, w, h = img_stack.shape
                if self.periph: # Fill the area with the blurred orignal
                    assert w % 4 == 0 and h % 4 == 0, ("Only resolutions divisible by 4"
                        " are supported")
                    # Create periphery with 1/16 of the original information
                    masked_state = np.stack(
                        [pyramid((img * 255).astype(np.uint8)[...,np.newaxis])[2]
                         for img in img_stack], dtype=np.float64)[...,0] / 255
                    # Visual info calculation
                    visual_info = float(((w * h) + 15 * (self.size[0] * self.size[1]))
                        / (16 * w * h))
                else: # Fill the area outside the crop with zeros
                    masked_state = np.full_like(img_stack, 0.5)
                    visual_info = float((self.size[0] * self.size[1]) / (w * h))

                for fixation in fixations:
                    crop = img_stack[
                        ...,
                        fixation[1]:fixation[1] + self.size[1],
                        fixation[0]:fixation[0] + self.size[0],
                    ]
                    masked_state[
                        ...,
                        fixation[1]:fixation[1] + self.size[1],
                        fixation[0]:fixation[0] + self.size[0],
                    ] = crop
                return masked_state, visual_info

            case "gaussian":
                raise NotImplementedError()
                gaussians = [None] * len(fixations)
                for i, fixation in enumerate(fixations):
                    distances_from_fov = self._pixel_eccentricities(
                    img_stack.shape[-2:], fixation)
                    sigma = self.size[0] / 2
                    gaussian = np.exp(-np.sum(
                        np.square(distances_from_fov) / (2 * np.square(sigma)),
                        axis=0)) # -> [84,84]
                    gaussians[i] = gaussian
                gaussian = np.sum(np.stack(gaussians), axis=0)
                gaussian /= np.max(gaussian)
                return img_stack * gaussian

            case "exponential":
                visual_infos = np.empty(len(img_stack))
                for i, img in enumerate(img_stack):
                    # Scale images between -1 and 1
                    img = (img[...,None] * 255).astype(np.uint8)
                    img, resolutions = foveat_img(img.copy(), fixations)
                    visual_info = resolutions.mean().item()
                    img = (img[...,0]).astype(np.float64) / 255
                    if self.weighting:
                        # 0 weighted values are pushed towards 0.5
                        img = (img - 0.5) * resolutions + 0.5
                    img_stack[i] = img
                    visual_infos[i] = visual_info
                visual_info = visual_infos.mean().item()
                return img_stack, visual_info

    def draw(self, frames: np.ndarray, fov_locs: np.ndarray):
        """ Draws the fovea onto a stack of frames

        :param Array[N,256,256,3] frames:
        :param Array[N,2] fov_locs:
        """
        color = [0,255,0] # Green
        match(self.type):
            # Draw the window for a windowed fovea
            case "window":
                for i in range(len(frames)):
                    top_left = fov_locs[i].astype(np.int32)
                    bottom_right = (top_left + self.size).astype(np.int32)
                    frames[i] = cv2.rectangle(frames[i], top_left, bottom_right,
                                              color, 1)
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

    @staticmethod
    def _resolution_mask(shape: tuple[int, int], fixations: list[tuple[int, int]]):
        """ Get a greyscale image indicating how unblurred every pixel is """
        pixel_eccentricities = Fovea._pixel_eccentricities(
            shape, fixations) # -> [N,2,84,84]
        # Convert from pixels to visual degrees
        eccentricities = (pixel_eccentricities.transpose(0,2,3,1)
            * VISUAL_DEGREES_PER_PIXEL).transpose(0,3,1,2)
        # Absolute 1D distances
        abs_distances = np.sqrt(np.square(eccentricities).sum(axis=1)) # -> [N,84,84]

        mask = A * np.exp(B * abs_distances) + C
        mask = mask.max(axis=0) # -> [84,84]
        mask /= mask.max()
        return mask
