import json

# Load both JSON files
with open('pill_labelling/labels/auto_labels_cleaned.json', 'r') as f:
    auto_labels = json.load(f)

with open('pill_labelling/labels/imprint_labels_vetted.json', 'r') as f:
    vetted_labels = json.load(f)

# Create a dictionary to track images by filename
images_dict = {}

# Add vetted labels first (higher priority)
for item in vetted_labels:
    images_dict[item['image']] = item

# Add auto labels (only if image not already in vetted set)
for item in auto_labels:
    if item['image'] not in images_dict:
        # Remove confidence field from auto labels for consistency
        cleaned_item = {
            'image': item['image'],
            'labels': [{k: v for k, v in label.items() if k != 'confidence'}
                       for label in item['labels']]
        }
        images_dict[item['image']] = cleaned_item

# Convert back to list and sort by image name for consistency
merged = sorted(images_dict.values(), key=lambda x: x['image'])

# Save merged result
with open('pill_labelling/labels/imprint_labels_expanded.json', 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged labels created:")
print(f"  Vetted labels: {len(vetted_labels)}")
print(f"  Auto labels: {len(auto_labels)}")
print(f"  Total unique images: {len(merged)}")
print(f"  New images from auto labels: {len(merged) - len(vetted_labels)}")
