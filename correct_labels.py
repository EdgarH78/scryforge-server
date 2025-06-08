import os
import json

INPUT_DIR = "./training/labels"         # Folder with original annotations
OUTPUT_DIR = "./training/labels_corrected"  # Folder to save corrected files
os.makedirs(OUTPUT_DIR, exist_ok=True)

def correct_bbox(annotation):
    # Modify all bounding box entries
    for result in annotation.get("annotations", []) + annotation.get("completions", []):
        for item in result.get("result", []):
            if item["type"] == "rectanglelabels":
                value = item["value"]
                x = value["x"]
                y = value["y"]
                w = value["width"]
                h = value["height"]

                # Convert from center to top-left
                value["x"] = x - w / 2
                value["y"] = y - h / 2
    return annotation

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected = correct_bbox(data)

    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2)

print("✅ Correction complete.")
