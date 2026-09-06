import cv2
import os
import yaml

# --------------------------------------------------
# 1. Find the project folder
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------
# 2. Dataset paths
# --------------------------------------------------

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "garbage_dataset",
    "train",
    "images"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "garbage_dataset",
    "train",
    "labels"
)

DATA_YAML = os.path.join(
    BASE_DIR,
    "datasets",
    "garbage_dataset",
    "data.yaml"
)


# --------------------------------------------------
# 3. Read class names from data.yaml
# --------------------------------------------------

with open(DATA_YAML, "r") as file:
    data = yaml.safe_load(file)

class_names = data["names"]


# --------------------------------------------------
# 4. Get all image files
# --------------------------------------------------

image_files = os.listdir(IMAGE_PATH)

print("Number of images:", len(image_files))
print("First few images:", image_files[:5])


# --------------------------------------------------
# 5. Select the first image
# --------------------------------------------------

image_name = image_files[0]

image_path = os.path.join(
    IMAGE_PATH,
    image_name
)


# --------------------------------------------------
# 6. Find the matching label file
# --------------------------------------------------

label_name = os.path.splitext(image_name)[0] + ".txt"

label_path = os.path.join(
    LABEL_PATH,
    label_name
)


# --------------------------------------------------
# 7. Read the image
# --------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {image_path}"
    )


# --------------------------------------------------
# 8. Get image dimensions
# --------------------------------------------------

image_height, image_width = image.shape[:2]

print("Image:", image_name)
print("Image width:", image_width)
print("Image height:", image_height)


# --------------------------------------------------
# 9. Read YOLO annotations
# --------------------------------------------------

if not os.path.exists(label_path):
    raise FileNotFoundError(
        f"Label file not found: {label_path}"
    )

with open(label_path, "r") as file:
    labels = file.readlines()


# --------------------------------------------------
# 10. Process every object in the image
# --------------------------------------------------

for label in labels:

    # Remove extra spaces/newline
    parts = label.strip().split()

    # Ignore empty lines
    if not parts:
        continue

    # YOLO format:
    # class_id x_center y_center width height

    class_id = int(parts[0])

    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])


    # --------------------------------------------------
    # 11. Convert normalized coordinates to pixels
    # --------------------------------------------------

    x_center = x_center * image_width
    y_center = y_center * image_height

    width = width * image_width
    height = height * image_height


    # --------------------------------------------------
    # 12. Convert center coordinates to corners
    # --------------------------------------------------

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)

    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)


    # --------------------------------------------------
    # 13. Get class name
    # --------------------------------------------------

    class_name = class_names[class_id]


    # --------------------------------------------------
    # 14. Draw bounding box
    # --------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # 15. Draw class name
    # --------------------------------------------------

    cv2.putText(
        image,
        class_name,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # 16. Print annotation information
    # --------------------------------------------------

    print(
        f"Class: {class_id} ({class_name}) | "
        f"Box: ({x1}, {y1}) -> ({x2}, {y2})"
    )


# --------------------------------------------------
# 17. Display the image
# --------------------------------------------------

cv2.imshow("Garbage Dataset Annotation", image)

cv2.waitKey(0)

cv2.destroyAllWindows()