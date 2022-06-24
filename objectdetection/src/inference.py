import numpy as np
import cv2
import torch
import glob as glob
from model import create_model
from config import CLASSES
import json
import time




# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '../outputs/model1.pth', map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = '../../rectified_img'

test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.7
path2 = '/home/fiodccob/GEO2020/regularization/3DFacade/objectdetection/pre_txt/'

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    time_start = time.time()
    with torch.no_grad():
        outputs = model(image)
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    f = open(path2 + image_name + ".txt", 'a')
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()


        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        labels = labels[scores >= detection_threshold].astype(np.int32)
        scorelist = scores[scores>=detection_threshold].astype(np.float32)
        # boxlist = boxes
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in labels]
        # print(len(pred_classes))
        # print(len(draw_boxes))
        dic = {}
        dic['boxes'] = draw_boxes.tolist()
        dic['labels'] = labels.tolist()
        newlist = []
        for l in dic['labels']:
            if(l == 1):
                newlist.append(0)
            elif(l == 3):
                newlist.append(1)
            else:
                newlist.append(2)
        dic['labels'] = newlist
        #
        for j, a in enumerate(draw_boxes):
            string = pred_classes[j]+' '+str(scorelist[j])+' '+str(a[0])+' '+str(a[1])+' '+str(a[2])+' '+str(a[3])+'\n'
            f.write(string)
         # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 3)
            # cv2.putText(orig_image, pred_classes[j]+str(scores[j]),
            #             (int(box[0]), int(box[1] - 5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            #             1)
        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        cv2.imwrite(f"../predictions/{image_name}.jpg", orig_image, )
        with open(f"../predictions/{image_name}.json", 'w') as fp:
            json.dump(dic, fp)

    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')

cv2.destroyAllWindows()