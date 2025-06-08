import os
import json

def update_label_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract filename from the original path
    original_path = data['data']['image']
    filename = os.path.basename(original_path.split("=", 1)[-1])  # handles ?d=path/to/file.jpg
    
    # Update to use full URL
    data['data']['image'] = f'http://localhost:5000/images/{filename}'
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    labels_dir = "training/labels"
    for filename in os.listdir(labels_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(labels_dir, filename)
            print(f"Updating {file_path}")
            update_label_file(file_path)
            print(f"✅ Updated {file_path}")

if __name__ == "__main__":
    main()
