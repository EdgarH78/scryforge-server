import os
import xml.etree.ElementTree as ET
from collections import Counter

label_counts = Counter()
bad_labels = set()
valid_labels = {"red", "pink", "yellow", "violet", "green", "orange", "blue"}

for filename in os.listdir("dataset/annotations"):
    if not filename.endswith(".xml"):
        continue
    path = os.path.join("dataset/annotations", filename)
    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        label_counts[name] += 1
        if name not in valid_labels:
            bad_labels.add((filename, name))

print("✅ Label counts:", label_counts)
if bad_labels:
    print("🚨 Bad labels found:")
    for fn, label in bad_labels:
        print(f"  {fn}: '{label}' (not in allowed set)")
else:
    print("🎉 No bad labels detected.")