import os
import shutil

def split_dataset(src_dir, dest_dir, num_subjects=10):
    """
    Splits first `num_subjects` folders into train and test sets.
    Test = first 10 images, Train = last 10 images.
    Saves results to `dest_dir/train` and `dest_dir/test`.
    """
    # Define target train & test folders
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get first `num_subjects` folders sorted (001, 002, ...)
    subject_folders = sorted(os.listdir(src_dir))[:num_subjects]

    for subject in subject_folders:
        subject_path = os.path.join(src_dir, subject)

        if not os.path.isdir(subject_path):
            continue

        # Create subject-specific folders in train & test
        train_subject_dir = os.path.join(train_dir, subject)
        test_subject_dir = os.path.join(test_dir, subject)
        os.makedirs(train_subject_dir, exist_ok=True)
        os.makedirs(test_subject_dir, exist_ok=True)

        # Sort image files
        images = sorted(os.listdir(subject_path))

        # First 10 images -> test, Last 10 images -> train
        test_images = images[:10]
        train_images = images[-10:]

        # Copy images
        for img in test_images:
            shutil.copy(
                os.path.join(subject_path, img),
                os.path.join(test_subject_dir, img)
            )

        for img in train_images:
            shutil.copy(
                os.path.join(subject_path, img),
                os.path.join(train_subject_dir, img)
            )

        print(f"Processed Subject: {subject}")

# === Usage ===
source_directory = "/home/aryan/Desktop/Adl_assignment_1/masked_dataset/test"
destination_directory = "/home/aryan/Desktop/Adl_assignment_1/masked_split"

split_dataset(source_directory, destination_directory, num_subjects=300)
