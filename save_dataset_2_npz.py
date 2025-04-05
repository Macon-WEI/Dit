import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_npz_from_train_folder(train_dir):
    """
    Builds a single .npz file from a folder containing multiple subfolders of images.

    Args:
    - train_dir (str): Path to the main training directory containing subfolders.

    Returns:
    - npz_path (str): Path to the saved .npz file.
    """
    samples = []
    class_folders = sorted(glob.glob(os.path.join(train_dir, '*')))


    for class_folder in tqdm(class_folders, desc="Building .npz file from samples"):
        class_samples = []
        image_files = sorted(glob.glob(os.path.join(class_folder, '*.JPEG')) +
                             glob.glob(os.path.join(class_folder, '*.jpg')) +
                             glob.glob(os.path.join(class_folder, '*.png')))

        # Take up to num_samples_per_class samples from each class folder
        for i in range(len(image_files)):
            image_file = image_files[i]
            sample_pil = Image.open(image_file)
            if i==0:
                print(sample_pil.size)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            class_samples.append(sample_np)

        samples.extend(class_samples)
        print(len(samples))

    samples = np.stack(samples)
    npz_path = os.path.join(train_dir, "all_samples.npz")
    print("--------------------npz_path000000000000000000000000",npz_path)
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

    return npz_path




def create_npz_from_train_folder_2(train_dir):
    class_folders = sorted(glob.glob(os.path.join(train_dir, '*')))
    sample_shapes = []
    total_samples = 0

    # First pass to count total samples and get the shape of each sample
    for class_folder in tqdm(class_folders, desc="Counting samples"):
        image_files = sorted(glob.glob(os.path.join(class_folder, '*.JPEG')) +
                             glob.glob(os.path.join(class_folder, '*.jpg')) +
                             glob.glob(os.path.join(class_folder, '*.png')))
        total_samples += len(image_files)
        if len(image_files) > 0:
            sample_pil = Image.open(image_files[0])
            sample_shapes.append(np.asarray(sample_pil).shape)

    # Assuming all images have the same shape, use the first image's shape
    if len(sample_shapes) > 0:
        sample_shape = sample_shapes[0]
    else:
        print("No images found in the provided directory.")
        return None

    print("here is all sample shape",sample_shapes)

    npz_path = os.path.join(train_dir, "all_samples.npz")
    print("-----------------------------npz_path000000000111111111111111",npz_path)
    # Use memmap to create an array on disk
    samples = np.memmap(npz_path, dtype='uint8', mode='w+', shape=(total_samples, *sample_shape))

    sample_index = 0

    # Second pass to write data into the memmap array
    for class_folder in tqdm(class_folders, desc="Writing samples to .npz"):
        image_files = sorted(glob.glob(os.path.join(class_folder, '*.JPEG')) +
                             glob.glob(os.path.join(class_folder, '*.jpg')) +
                             glob.glob(os.path.join(class_folder, '*.png')))
        for image_file in image_files:
            sample_pil = Image.open(image_file)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples[sample_index] = sample_np
            sample_index += 1

    # Flush memmap array to disk
    samples.flush()

    # Save the memmap array to .npz file
    np.savez(npz_path, arr_0=samples)

    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

    return npz_path


def create_npz_from_train_folder_3(train_dir, target_size=(256, 256)):
    class_folders = sorted(glob.glob(os.path.join(train_dir, '*')))
    total_samples = 0

    # First pass to count total samples
    for class_folder in tqdm(class_folders, desc="Counting samples"):
        image_files = sorted(glob.glob(os.path.join(class_folder, '*.JPEG')) +
                             glob.glob(os.path.join(class_folder, '*.jpg')) +
                             glob.glob(os.path.join(class_folder, '*.png')))
        total_samples += len(image_files)

    sample_shape = (*target_size, 3)

    npz_path = os.path.join(train_dir, "all_samples.npz")
    print("--------------npz_path111111111111111111-------",npz_path)
    # Use memmap to create an array on disk
    samples = np.memmap(npz_path, dtype='uint8', mode='w+', shape=(total_samples, *sample_shape))

    sample_index = 0

    # Second pass to write data into the memmap array
    for class_folder in tqdm(class_folders, desc="Writing samples to .npz"):
        image_files = sorted(glob.glob(os.path.join(class_folder, '*.JPEG')) +
                             glob.glob(os.path.join(class_folder, '*.jpg')) +
                             glob.glob(os.path.join(class_folder, '*.png')))
        for image_file in image_files:
            sample_pil = Image.open(image_file)
            sample_pil = sample_pil.resize(target_size).convert('RGB')  # Convert image to RGB
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples[sample_index] = sample_np
            sample_index += 1

    # Flush memmap array to disk
    samples.flush()
    print("samples.flush finish")
    # Save the memmap array to .npz file
    #np.savez(npz_path, samples=samples)
    #np.savez_compressed(npz_path,samples=samples)
    np.save(npz_path, samples=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

    return npz_path


# Example usage:
npz_file = create_npz_from_train_folder_3('/public/home/acr0vd9ik6/project/DiT/train')
