import torch
import torchvision
import numpy as np
import cv2
from model import create_model
import glob as glob


# set the computation device
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# load the model and the trained weights
model = create_model(num_classes=4)
model.load_state_dict(torch.load(
    '../outputs2/model100.pth'))
model.eval()

for param in model.parameters():
    print(param.device)


DIR_TEST = '../../rectified'
test_images = glob.glob(f"{DIR_TEST}/*")

image_name = test_images[0].split('/')[-1].split('.')[0]
image = cv2.imread(test_images[0])

height = image.shape[0]
width = image.shape[1]

orig_image = image.copy()
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
image /= 255.0
    # bring color channels to front
image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
image = torch.unsqueeze(image, 0)

example = torch.rand(1,3,224,224)

traced_script_module = torch.jit.trace(model,image)
