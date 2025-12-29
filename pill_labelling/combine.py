import json

# Load the two JSON files
with open('labels_final/imprint_labels_batch_2.json', 'r') as f1:
    labels1 = json.load(f1)

with open('labels_final/imprint_labels_batch_3_vetted.json', 'r') as f2:
    labels2 = json.load(f2)

# Create a dictionary to store unique images (using image name as key)
combined_dict = {}

# Add items from first file
for item in labels1:
    image_name = item['image']
    combined_dict[image_name] = item

# Add items from second file (this will overwrite duplicates with the second file's version)
for item in labels2:
    image_name = item['image']
    combined_dict[image_name] = item

# Convert back to list
combined_labels = list(combined_dict.values())

# Save to new file
with open('labels_final/imprint_labels_batch_3.json', 'w') as f_out:
    json.dump(combined_labels, f_out, indent=2)

print(f"Combined {len(labels1)} labels from file 1 and {len(labels2)} labels from file 2")
print(f"Total unique images: {len(combined_labels)}")
print(f"Output saved to: pill_labelling/labels_final/imprint_labels_batch_3.json")
