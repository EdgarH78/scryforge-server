import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
import os
import cv2
import uuid

# Define paths
image_dir = "dataset/images"
annotation_dir = "dataset/annotations"
aug_image_dir = "dataset/augmented/images"
aug_annotation_dir = "dataset/augmented/annotations"

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_annotation_dir, exist_ok=True)

# Albumentations transform
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.4)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def parse_pascal_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return image_filename, width, height, boxes, labels, tree

def create_pascal_voc_xml(base_tree, filename, boxes, labels, width, height, save_path):
    root = base_tree.getroot()
    root.find('filename').text = filename
    root.find('size').find('width').text = str(width)
    root.find('size').find('height').text = str(height)

    for obj in root.findall('object'):
        root.remove(obj)

    for label, box in zip(labels, boxes):
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin, xmax, ymax = map(int, box)
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    ET.ElementTree(root).write(save_path)

augmented_count = 0

for xml_file in os.listdir(annotation_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(annotation_dir, xml_file)
    try:
        image_filename, width, height, boxes, labels, tree = parse_pascal_voc(xml_path)
    except:
        continue

    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    if image is None or not boxes:
        continue

    try:
        augmented = transform(image=image, bboxes=boxes, class_labels=labels)
    except:
        continue

    new_filename = f"{uuid.uuid4().hex}.jpg"
    new_image_path = os.path.join(aug_image_dir, new_filename)
    new_xml_path = os.path.join(aug_annotation_dir, new_filename.replace(".jpg", ".xml"))

    cv2.imwrite(new_image_path, augmented['image'])
    create_pascal_voc_xml(tree, new_filename, augmented['bboxes'], labels, width, height, new_xml_path)
    augmented_count += 1

print(f"✅ Augmented {augmented_count} new samples.")
