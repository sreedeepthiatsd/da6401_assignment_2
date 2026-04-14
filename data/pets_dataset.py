"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, images_dir, annotations_file,xml_dir=None, transform=None):
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.xml_dir = xml_dir
        self.transform = transform
        # to store samples
        self.samples = []
        # reading annotation file
        with open(self.annotations_file, 'r') as f :
            for line in f:
                # to handle blank lines
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                image_id = parts[0]  # first element 
                class_id = int(parts[1]) - 1 # we are converting to 0 based indexing


                img_path = os.path.join(self.images_dir, image_id + ".jpg")
                xml_path = os.path.join(self.xml_dir, image_id + ".xml")

                if os.path.exists(img_path) and os.path.exists(xml_path) :
                  self.samples.append({
                      "image_id": image_id,
                      "label": class_id
                  })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        
        sample = self.samples[idx]
        image_id = sample["image_id"]
        label = sample["label"]

        # construct the image path
        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        # check if the file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        # load the image from the dataset
        image = Image.open(img_path).convert("RGB")

        # load bounding box
        bbox = None

        xml_path = os.path.join(self.xml_dir, image_id + ".xml")
        

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # convert to center format
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        img_width, img_height = image.size

        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
        
        
        mask_path = os.path.join(self.images_dir.replace("images", "annotations/trimaps"), image_id + ".png")
        # --- MASK ---
        mask = Image.open(mask_path)
        mask = mask.resize((224, 224), Image.NEAREST)

        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask == 1).long()
        
        # apply the transforms
        if self.transform:
          image = self.transform(image)
        

        return {
            "image": image,
            "label": label,
            "image_id": image_id,
            "bbox": bbox,
            "mask": mask    
        }
