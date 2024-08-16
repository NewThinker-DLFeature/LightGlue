#!/usr/bin/env python3

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, frame2tensor, AverageTimer, VideoStreamer
from lightglue import match_pair
from pathlib import Path
import cv2
import numpy as np
import time
import argparse
import torch

import matplotlib.pyplot as plt


# Draw matches
def draw_matches(image0_np, image1_np, points0_np, points1_np):
  # Create a new output image that concatenates the two images together
  h0, w0, _ = image0_np.shape
  h1, w1, _ = image1_np.shape
  output_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
  output_img[:h0, :w0] = image0_np
  output_img[:h1, w0:] = image1_np

  # Draw lines between matching points
  for (x0, y0), (x1, y1) in zip(points0_np, points1_np):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.line(output_img, (int(x0), int(y0)), (int(x1) + w0, int(y1)), color, 1)
    cv2.circle(output_img, (int(x0), int(y0)), 2, color, -1)
    cv2.circle(output_img, (int(x1) + w0, int(y1)), 2, color, -1)
  
  cv2.putText(output_img, "LightGlue", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  cv2.putText(output_img, "num of matches: {}".format(len(points0_np)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

  return output_img

def to_np_image(image):
  image_np = image.cpu().numpy().transpose(1, 2, 0)

  # Convert from [0,1] to [0,255] for display
  image_np = (image_np * 255).astype(np.uint8)

  # BGR to RGB
  image_np = image_np[..., ::-1]

  return image_np

def to_np_points(points):
  return points.cpu().numpy()


if __name__ == '__main__':
  import sys
  parser = argparse.ArgumentParser(
      description='LightGlue demo',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--input', type=str, default='0',
      help='ID of a USB webcam, URL of an IP camera, '
            'or path to an image directory or movie file')
  parser.add_argument(
      '--output_dir', type=str, default=None,
      help='Directory where to write output frames (If None, no output)')
  parser.add_argument(
      '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg', '*.JPG'],
      help='Glob if a directory of images is specified')
  parser.add_argument(
      '--skip', type=int, default=1,
      help='Images to skip if input is a movie or directory')
  parser.add_argument(
      '--max_length', type=int, default=1000000,
      help='Maximum length if input is a movie or directory')
  parser.add_argument(
      '--resize', type=int, nargs='+', default=[640, 480],
      help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')

  parser.add_argument(
      '--show_keypoints', action='store_true',
      help='Show the detected keypoints')
  parser.add_argument(
      '--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely')
  parser.add_argument(
      '--force_cpu', action='store_true',
      help='Force pytorch to run in CPU mode.')

  opt = parser.parse_args()
  print(opt)

  device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
  print('Running inference on device \"{}\"'.format(device))

  if len(opt.resize) == 2:
    print('Will resize to {}x{} (WxH)'.format(
        opt.resize[0], opt.resize[1]))
  elif len(opt.resize) == 1 and opt.resize[0] > 0:
    print('Will resize max dimension to {}'.format(opt.resize[0]))
    opt.resize = opt.resize[0]
  elif len(opt.resize) == 1:
    print('Will not resize images')
    opt.resize = None
  else:
    raise ValueError('Cannot specify more than two integers for --resize')

  # SuperPoint+LightGlue
  extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
  matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

  # # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
  # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
  # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

  vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                      opt.image_glob, opt.max_length)
  frame, ret = vs.next_frame()
  assert ret, 'Error when reading the first frame (try different --input?)'

  if not opt.no_display:
      cv2.namedWindow('LightGlue matches', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('LightGlue matches', 640*2, 480)
  else:
      print('Skipping visualization, will not show a GUI.')


  # extract local features
  frame = frame.to(device)
  extracted_feats0 = extractor.extract(frame)  # auto-resize the image, disable with resize=None
  image0_np = to_np_image(frame)
  image0_id = 0


  timer = AverageTimer()


  while True:
    frame, ret = vs.next_frame()
    if not ret:
      print('Finished demo_lightglue.py')
      break
    timer.update('data')
    stem0, stem1 = image0_id, vs.i - 1

    frame = frame.to(device)
    extracted_feats1 = extractor.extract(frame)  # auto-resize the image, disable with resize=None
    timer.update('extract')


    # match the features
    matches01 = matcher({'image0': extracted_feats0, 'image1': extracted_feats1})
    timer.update('match')

    feats0, feats1, matches01 = [rbd(x) for x in [extracted_feats0, extracted_feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)

    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points0_np = to_np_points(points0)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    points1_np = to_np_points(points1)
    image1_np = to_np_image(frame)

    matched_cv_img = draw_matches(image0_np, image1_np, points0_np, points1_np)

    if not opt.no_display:
      cv2.imshow('LightGlue matches', matched_cv_img)
      key = chr(cv2.waitKey(1) & 0xFF)
      if key == 'q':
        vs.cleanup()
        print('Exiting (via q) demo_lightglue.py')
        break
      elif key == 'n':  # set the current frame as anchor
        extracted_feats0 = extractor.extract(frame)  # auto-resize the image, disable with resize=None
        image0_np = to_np_image(frame)
        image0_id = (vs.i - 1)
      elif key == 'k':
        opt.show_keypoints = not opt.show_keypoints
      elif key == ' ':
        # Pause on spacebar press.
        # print('\nPaused. Press space to continue.')
        while True:
            key = chr(cv2.waitKey(0) & 0xFF)
            if key == ' ':
                break

    timer.update('viz')
    timer.print()

    if opt.output_dir is not None:
      #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
      stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
      out_file = str(Path(opt.output_dir, stem + '.png'))
      print('\nWriting image to {}'.format(out_file))
      cv2.imwrite(out_file, matched_cv_img)

  cv2.destroyAllWindows()
  vs.cleanup()

