import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset.txt from images folder"
    )
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument(
        "--images", type=str, help="Relative path to images folder inside dataset"
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="Number of images to select (default: 1000)",
    )
    args = parser.parse_args()

    dataset_folder = args.dataset
    images_rel = args.images
    images_folder = os.path.join(dataset_folder, images_rel)

    # List all files in images_folder
    all_files = [
        f
        for f in os.listdir(images_folder)
        if os.path.isfile(os.path.join(images_folder, f))
    ]
    if len(all_files) < args.number:
        print(f"Warning: Only {len(all_files)} images found, less than {args.number}.")
        selected = all_files
    else:
        selected = random.sample(all_files, args.number)
    selected_rel_paths = [
        os.path.join(images_rel, f).replace(os.sep, "/") for f in selected
    ]

    dataset_txt_path = os.path.join(dataset_folder, "dataset.txt")
    with open(dataset_txt_path, "w") as f:
        for rel_path in selected_rel_paths:
            f.write(rel_path + "\n")
    print(f"Saved {len(selected_rel_paths)} image paths to {dataset_txt_path}")


if __name__ == "__main__":
    main()
