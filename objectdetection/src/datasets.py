import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
import re
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform


# the dataset class
class FacadeDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        with open(annot_file_path,mode='r+') as f:
            lines = f.readlines()
            if lines[0] == '<object>\n':
                f.seek(0, 0)
                content = f.read()
                f.seek(0,0)
                f.write('<root>\n'+content)

        with open(annot_file_path,mode='r+') as f:
            lines = f.readlines()
            if lines[-1] == '</object>\n':
                f.write("</root>")
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            if member.find('labelname').text == '\nwindow\n' or member.find('labelname').text == '\ndoor\n' or member.find('labelname').text == '\nbalcony\n':
                labels.append(self.classes.index(member.find('labelname').text.replace('\n','')))
                # xmin = left corner x-coordinates
                ymin = int(float(member.find('points').findall('x')[0].text.replace('\n','')) * image_height)
                # xmax = right corner x-coordinates
                ymax = int(float(member.find('points').findall('x')[1].text.replace('\n','')) * image_height)
                # ymin = left corner y-coordinates
                xmin = int(float(member.find('points').findall('y')[0].text.replace('\n','')) * image_width)
                # ymax = right corner y-coordinates
                xmax = int(float(member.find('points').findall('y')[1].text.replace('\n','')) * image_width)

                # resize the bounding boxes according to the...
                # ... desired `width`, `height`
                xmin_final = (xmin / image_width) * self.width
                xmax_final = (xmax / image_width) * self.width
                ymin_final = (ymin / image_height) * self.height
                yamx_final = (ymax / image_height) * self.height

                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
train_dataset = FacadeDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = FacadeDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = FacadeDataset(
        VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")


    # function to visualize a single sample
    def visualize_sample(image, target):

        for idx,i in enumerate(target['boxes']):
            box = target['boxes'][idx]
            label = CLASSES[target['labels'][idx]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 20
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]

        visualize_sample(image,target)