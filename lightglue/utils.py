import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import time
from collections import OrderedDict
from threading import Thread

import cv2
import kornia
import numpy as np
import torch


class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats


def match_pair(
    extractor,
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: str = "cpu",
    **preprocess,
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    feats0 = extractor.extract(image0, **preprocess)
    feats1 = extractor.extract(image1, **preprocess)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()

class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:
            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
    
            if self.resize is not None:
                image = resize_image(image, self.resize)
            image = numpy_image_to_torch(image)
            # print(" self.camera: image type {}, image shape {}".format(type(image), image.shape))
        else:
            image_file = str(self.listing[self.i])
            image = load_image(image_file, self.resize)
            # print("!self.camera: image type {}, image shape {}".format(type(image), image.shape))
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False
