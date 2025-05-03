import os
import shutil
import random
from pathlib import Path

# Create new split folders with Train / Test / Validation --------------------------------------------------------------------
def move_images(base_dir, source_dir, dest_dir, file_counts, seed=42):
    random.seed(seed)
    for subdir, num_to_move in file_counts.items():
        source_subdir = os.path.join(base_dir, source_dir, subdir)
        dest_subdir = os.path.join(base_dir, dest_dir, subdir)
        os.makedirs(dest_subdir, exist_ok=True)

        images = [f for f in os.listdir(source_subdir) if os.path.isfile(os.path.join(source_subdir, f))]

        if num_to_move > len(images):
            raise ValueError(f"Requested {num_to_move} files from {source_subdir}, but only {len(images)} available.")

        selected_images = random.sample(images, num_to_move)

        print(f"Moving {num_to_move} images from {source_subdir} to {dest_subdir}")

        for img_name in selected_images:
            src_path = os.path.join(source_subdir, img_name)
            dest_path = os.path.join(dest_subdir, img_name)
            shutil.move(src_path, dest_path)

# Sanity Check ----------------------------------------------------------------------------------------
def sanity_check(base_dir, folders, file_counts):
    for folder in folders:
        for subdir in ["Train", "Test", "Validation"]:
            subdir_path = os.path.join(base_dir, folder, subdir)
            if not os.path.exists(subdir_path):
                raise ValueError(f"Subdirectory {subdir_path} does not exist.")
            if len(os.listdir(subdir_path)) == 0:
                raise ValueError(f"Subdirectory {subdir_path} is empty.")
            if len(os.listdir(subdir_path)) != file_counts[subdir]:
                raise ValueError(f"Subdirectory {subdir_path} has {len(os.listdir(subdir_path))} files, expected {file_counts[subdir]}.")
            else:
                print(f"Sanity check passed for {subdir_path} with {len(os.listdir(subdir_path))} files.")

# Merge folders ----------------------------------------------------------------------------------------
def merge_folders(base_dir, merge_source_folder, merge_destination_folder):
    count = 0
    merge_source_folder = os.path.join(base_dir, merge_source_folder)
    merge_destination_folder = os.path.join(base_dir, merge_destination_folder)
    merge_source_folder = Path(merge_source_folder)
    merge_destination_folder = Path(merge_destination_folder)
    if not merge_source_folder.exists():
        raise ValueError(f"Source folder {merge_source_folder} does not exist.")
    os.makedirs(merge_destination_folder, exist_ok=True)
    for root, dirs, files in os.walk(merge_source_folder):
        root_path = Path(root)
        relative_path = root_path.relative_to(merge_source_folder)
        dest_path = merge_destination_folder / relative_path
        dest_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = root_path / file
            dst_file = dest_path / file

            # Overwrite if exists
            shutil.copy2(src_file, dst_file)
            count += 1
    print(f"Total files moved: {count}")

if __name__ == "__main__":

    # copy real images randomly to the new split folders
    folders = ['Real_1_k_split', 'Real_2_k_split', 'Real_3_k_split', 'Real_4_k_split', 'Real_5_k_split']
    file_counts = {
        'Train': 700,
        'Test': 150,
        'Validation': 150
    }
    file_check_counts = {
        'Train': 700,
        'Test': 150,
        'Validation': 150
    }
    full_file_check_counts = {
        'Train': 2800,
        'Test': 600,
        'Validation': 600
    }
    base_dir = '/home/ec2-user/madde/datasets'

    # Copy 1st 1k folder
    merge_source_folder = "Real_1k_split"
    merge_destination_folder = "Real_1_k_split"
    merge_folders(base_dir, merge_source_folder, merge_destination_folder)
    sanity_check(base_dir, [merge_destination_folder], file_check_counts)
    merge_folders(base_dir, merge_source_folder, "Real_1-5_k_split")
    sanity_check(base_dir, ["Real_1-5_k_split"], file_check_counts)

    # Copy 4k splits
    merge_source_folder = "Real_4k_split"
    merge_destination_folder = "Real_4k_split_copy"
    merge_folders(base_dir, merge_source_folder, merge_destination_folder)
    sanity_check(base_dir, [merge_destination_folder], full_file_check_counts)

    # Create 1000 images datest for iterative retraining
    real_combination = ["Real_1-5_k_split", "Real_2-5_k_split", "Real_3-5_k_split", "Real_4-5_k_split", "Real_5-5_k_split"]


    # Move real images to the new split folders
    move_source_folder = "Real_4k_split_copy"
    for idx, folder in enumerate(folders[1:]):
        move_images(base_dir, move_source_folder, folder, file_counts)
        merge_folders(base_dir, folder, real_combination[idx+1])
        sanity_check(base_dir, [real_combination[idx+1]], file_counts)
        
        merge_folders(base_dir, folders[idx], folder)

        # Sanity check
        file_check_counts["Train"] += 700
        file_check_counts["Test"] += 150
        file_check_counts["Validation"] += 150
        sanity_check(base_dir, [folder], file_check_counts)
        


    