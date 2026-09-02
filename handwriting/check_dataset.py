import os

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def count_dataset(data_folder):
    counts = {}

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Convert s-a -> a, s-b -> b, etc.
        if folder_name.startswith("s-"):
            label = folder_name[2:]
        else:
            label = folder_name

        count = sum(
            1
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )

        counts[label] = counts.get(label, 0) + count

    return counts


if __name__ == "__main__":
    counts = count_dataset("data")

    print("Training Dataset Distribution\n")
    print("-" * 30)

    for label in sorted(counts):
        print(f"{label}: {counts[label]}")

    print("-" * 30)
    print(f"Total training images: {sum(counts.values())}")