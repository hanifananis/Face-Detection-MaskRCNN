# -*- coding: utf-8 -*-
import os
import json

dir = "D:/Hanifan/Face-Detection-Base-on-Mask-R-CNN-master/wider_face_split/wider_face_val"

for root, _, files in os.walk(dir):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'r') as txtfile:
                lines = txtfile.readlines()
                i = 0
                while i < len(lines):
                    # First line is the image path
                    image_path = lines[i].strip()
                    i += 1
                    if i >= len(lines):
                        break
                    # Second line might be the number of bounding boxes or something else
                    try:
                        num_boxes = int(lines[i].strip())
                        i += 1
                    except ValueError:
                        # If it's not a number, assume it's another image path and continue
                        continue
                    # Following lines are the bounding box coordinates
                    bounding_boxes = []
                    for _ in range(num_boxes):
                        if i >= len(lines):
                            break
                        box_coords = lines[i].strip().split()
                        if len(box_coords) < 4:
                            i += 1
                            continue
                        try:
                            x1 = int(box_coords[0])
                            y1 = int(box_coords[1])
                            width = int(box_coords[2])
                            height = int(box_coords[3])
                        except ValueError:
                            i += 1
                            continue
                        bounding_boxes.append({
                            'x': [x1, x1 + width, x1 + width, x1],
                            'y': [y1, y1, y1 + height, y1 + height]
                        })
                        i += 1
                    # Create a dictionary for the bounding boxes
                    dict_boxes = {idx: bbox for idx, bbox in enumerate(bounding_boxes)}
                    # Write the JSON file
                    json_filename = os.path.join(root, os.path.basename(image_path).replace('.jpg', '.json'))
                    with open(json_filename, 'w+') as jsonfile:
                        json.dump(dict_boxes, jsonfile, indent=4)
                    print(f"Created JSON for {image_path}")
