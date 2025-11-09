import os

# Specify directory paths
label_dir = "dataset/labels/train"
image_dir = "dataset/images/train"

# Labels to keep
valid_classes = {5}


def process_annotations():
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, os.path.splitext(label_file)[0] + ".jpg")

        # Read annotation file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Filter labels to keep
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 1:
                continue

            label = int(parts[0])
            if label in valid_classes:
                filtered_lines.append(line)

        if filtered_lines:
            # Keep needed labels and overwrite file
            with open(label_path, "w") as f:
                f.writelines(filtered_lines)
        else:
            # If no needed labels, delete label file and corresponding image
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            print(f"Deleted: {label_path}")


if __name__ == "__main__":
    process_annotations()
