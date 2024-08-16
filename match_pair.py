#!/usr/bin/env python3

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import match_pair
import cv2
import numpy as np
import time
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
  if len(sys.argv) < 3:
    print('Usage: python3 {}  []'.format(sys.argv[0]))
    print('Examples: ')
    print('- python3 {} assets/DSC_0410.JPG assets/DSC_0411.JPG'.format(sys.argv[0]))
    print('- python3 {} assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg'.format(sys.argv[0]))
    sys.exit(1)

  image0_path = sys.argv[1]
  image1_path = sys.argv[2]

  # SuperPoint+LightGlue
  extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
  matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher


  # # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
  # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
  # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

  # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
  image0 = load_image(image0_path).cuda()
  image1 = load_image(image1_path).cuda()

  # extract local features
  feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
  feats1 = extractor.extract(image1)

  # match the features
  matches01 = matcher({'image0': feats0, 'image1': feats1})
  feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

  # # We also provide a convenience method to match a pair of images:
  # feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)

  matches = matches01['matches']  # indices with shape (K,2)
  points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
  points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

  image0_np = to_np_image(image0)
  image1_np = to_np_image(image1)
  points0_np = to_np_points(points0)
  points1_np = to_np_points(points1)

  matched_cv_img = draw_matches(image0_np, image1_np, points0_np, points1_np)

  cv2.imshow('Matches', matched_cv_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
