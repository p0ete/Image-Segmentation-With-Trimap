import cv2
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

import os

ITERATIONS = 6
KERNEL_LEN = 8


DIM_MODEL = "models/BEST_checkpoint.tar"
MASKRCNN_MODEL = "models/mask_rcnn_coco.h5"
MODEL_DIR = "logs/"

input_path = "Data/input/"
trimap_path = "Data/trimap/"
output_path = "Data/output/"
alpha_matte_path = "Data/alpha/"
foreground_path = "Data/foreground/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    
def image_fill(img, size, value):
    border = [math.ceil((size[0] - img.shape[0]) / 2),
              math.floor((size[0] - img.shape[0]) / 2),
              math.ceil((size[1] - img.shape[1]) / 2),
              math.floor((size[1] - img.shape[1]) / 2)]
    return cv2.copyMakeBorder(img, border[0], border[1], border[2], border[3], cv2.BORDER_CONSTANT, value=value)


def generate_trimap(segment_mask, trimap_name):
    k_size = KERNEL_LEN
    iterations = ITERATIONS
    alpha = segment_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)

    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    cv2.imwrite(trimap_name, trimap)


"""
Deep Image Matting

Predict the Alpha matte based on the input image and trimap.
"""


def predict_alpha(input_image, trimap_image):
    checkpoint = torch.load(DIM_MODEL, map_location=torch.device('cpu'))
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']
    img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    trimap = cv2.imread(trimap_image, 0)

    h, w = img.shape[:2]

    x = torch.zeros((1, 4, h, w), dtype=torch.float)
    image = img[..., ::-1]  # RGB
    image = transforms.ToPILImage()(image)
    image = transformer(image)
    x[0:, 0:3, :, :] = image

    x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)

    # Move to GPU, if available
    x = x.type(torch.FloatTensor).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()
    pred = pred.reshape((h, w))

    pred[trimap == 0] = 0.0
    pred[trimap == 255] = 1.0

    out = (pred.copy() * 255).astype(np.uint8)

    return out


"""
Mask R-CNN by Matterport

Generate initial alpha mask from the instance segemented by Mask-RCNN as proposed in the paper.
"""


def predict_mask(image_name, trimap_name):
    print ("predicting mask")
    config = CocoConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(MASKRCNN_MODEL, by_name=True)
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    # Run detection

    if image_name == None:
        print(image_name + "does not exist.")
        return
    results = model.detect([image], verbose=0)

    r = results[0]

    masks = r['masks']
    if len(r['class_ids']) ==2:
        mask = (masks[:, :, 0] == 1) | (masks[:,:,1] == 1)
    elif len(r['class_ids'])  ==3:
        mask = (masks[:,:,0] == 1) | (masks[:,:,1] == 1) | (masks[:,:,2] == 1)
    elif len(r['class_ids']) ==4:
        mask = (masks[:,:,0] == 1) | (masks[:,:,1] == 1) | (masks[:,:,2] == 1) | (masks[:,:,3] == 1)
    else:
        mask = masks[:,:,0]
    
    mask = np.uint8(mask * 255)
    generate_trimap(mask, str(trimap_name))

        


"""
Run End to End Pipeline as Proposed in the paper.
"""


def predict_end_to_end(input_image):
    print ("++++++++++++++++++++++++++++++++++++ BG Removal ++++++++++++++++++++++++++++++++++")
    file = input_image.split(".")[0]
    input_image = input_path + input_image
    trimap_image = input_path + "trimap_" + file + ".jpg"

    predict_mask(input_image, trimap_image)
    alpha = predict_alpha(input_image, trimap_image)
    output_name = output_path + "output_stage[0]_" + str(file) + ".png"
    cv2.imwrite(output_name, alpha)

    # Feedback Stage
    for i in range(ITERATIONS, 1, -1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i - 1, i - 1))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        unknown = cv2.dilate(unknown, kernel, iterations=1)
        trimap = fg * 255 + (unknown - fg) * 128

        trimap_name = trimap_path + "trimap_stage[" + str(i) + "]_" + str(file) + ".jpg"
        cv2.imwrite(trimap_name, trimap)

        alpha = predict_alpha(input_image, trimap_name)

        output_name = output_path + "output_stage[" + str(i) + "]_" + str(file) + ".jpg"
        cv2.imwrite(output_name, alpha)

    alpha_matte_name = alpha_matte_path + "alpha_" + str(file) + ".png"
    cv2.imwrite(alpha_matte_name, alpha)

    foreground_name = foreground_path + str(file) + ".png"
    #foreground_name = foreground_path + str(ITERATIONS) +"_"+ str(KERNEL_LEN) + ".png"
    input = Image.open(input_image)
    alpha_matte = Image.open(alpha_matte_name).convert('L')

    input.putalpha(alpha_matte)
    input.save(foreground_name)


if __name__ == "__main__":

    KERNEL_LEN = 3
    ITERATIONS = 2
    for image in os.listdir(input_path):
        predict_end_to_end(image)
