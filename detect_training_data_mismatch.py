import os
from pathlib import Path

image_dir = Path("dataset/images")
annotation_dir = Path("dataset/annotations")

image_files = {f.stem for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
annotation_files = {f.stem for f in annotation_dir.glob("*.xml")}

missing_annotations = image_files - annotation_files
missing_images = annotation_files - image_files

print("🟡 Images without annotations:", missing_annotations)
print("🔴 Annotations without images:", missing_images)