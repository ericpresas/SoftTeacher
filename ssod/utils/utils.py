COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def plot_annotations(np_img, boxes=None):
    pil_img = Image.fromarray(np.uint8(np_img)).convert('RGB')
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if boxes is not None:
      for (x, y, w, h), c in zip(boxes, colors):
          ax.add_patch(plt.Rectangle((x, y), w, h,
                                    fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.show()

def plot_annotations_xyxy(np_img, boxes=None):
    pil_img = Image.fromarray(np.uint8(np_img)).convert('RGB')
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if boxes is not None:
      for (x, y, x1, y1), c in zip(boxes, colors):
          w = x1 - x
          h = y1 - y
          ax.add_patch(plt.Rectangle((x, y), w, h,
                                    fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.show()
