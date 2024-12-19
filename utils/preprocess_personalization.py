import os

import imageio
import numpy as np
from PIL import Image


def concatenate_images_and_masks(images, masks, angle_output_dir, num_of_obj):
    # Ensure we have 4 images and 4 masks
    if num_of_obj == 3:
        images.append(Image.new('RGBA', (256, 256), (255, 255, 255, 0)))  # Transparent placeholder
        masks.append(Image.new('L', (256, 256), 0))  # Black mask for placeholder

    # Create a new image and masks of the required size with transparency
    combined_img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))  # Transparent background
    num_of_masks = num_of_obj if num_of_obj == 2 else 4
    combined_masks = [Image.new('L', (512, 512), 0) for _ in range(num_of_masks)]

    # Position of each image/mask in the combined image/mask
    if num_of_obj == 2:
        positions = [(0, 0), (256, 0)]
    else:
        positions = [(0, 0), (256, 0), (0, 256), (256, 256)]  # Horizontal arrangement

    for img, mask, pos in zip(images, masks, positions):
        combined_img.paste(img, pos, img)  # Use img as mask for itself to handle transparency
        combined_masks[positions.index(pos)].paste(mask, pos)

    # Save the combined image and masks
    os.makedirs(angle_output_dir, exist_ok=True)
    combined_img.save(os.path.join(angle_output_dir, "img.png"), "PNG")  # Save as PNG to maintain transparency
    for i, mask in enumerate(combined_masks):
        mask.save(os.path.join(angle_output_dir, f"mask{i}.png"), "PNG")


def find_angles(obj_dir):
    # Find all angles available for an object
    angles = []
    for filename in os.listdir(obj_dir):
        if filename.endswith('.png') and not filename.endswith('_mask.png'):
            # Extract the angle part from the filename
            angle = '_'.join(filename.split('_')[1:-1])
            angles.append(angle)
    return angles


def find_bounding_box(alpha):
    """ Find the bounding box of the non-transparent area in the alpha channel. """
    arr = np.array(alpha)
    x, y = np.where(arr != 0)
    return min(x) - 2, max(x) + 2, min(y) - 2, max(y) + 2


def adjust_aspect_ratio(bbox, aspect_ratio=0.5):  # aspect_ratio = width / height
    """ Adjust the bounding box to the specified aspect ratio (width:height). """
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    current_aspect_ratio = width / height

    if current_aspect_ratio > aspect_ratio:
        # Width is too large, adjust height
        extra_height = width / aspect_ratio - height
        y_min -= extra_height / 2
        y_max += extra_height / 2
    elif current_aspect_ratio < aspect_ratio:
        # Height is too large, adjust width
        extra_width = height * aspect_ratio - width
        x_min -= extra_width / 2
        x_max += extra_width / 2

    # Ensure coordinates are within image bounds and are integers
    x_min, x_max = max(0, int(x_min)), min(2048, int(x_max))
    y_min, y_max = max(0, int(y_min)), min(2048, int(y_max))

    return x_min, x_max, y_min, y_max


def crop_and_resize(image, bbox, size=(256, 256)):
    """ Crop the image to the bounding box and resize it. """
    if isinstance(image, np.ndarray):
        # If the image is a NumPy array, convert it to a PIL Image
        image = Image.fromarray(image)

    cropped_image = image.crop((bbox[2], bbox[0], bbox[3], bbox[1]))  # PIL uses (left, upper, right, lower)
    resized_image = cropped_image.resize(size, Image.Resampling.LANCZOS)
    return resized_image


def get_angle_filenames(obj_dir, angle):
    # find the filenames for the specified angle
    for filename in os.listdir(obj_dir):
        if angle in filename:
            return filename


def center_within_combined_bbox(original_bbox, combined_width, combined_height):
    """ Center the original bounding box within the combined bounding box dimensions. """
    orig_x_min, orig_x_max, orig_y_min, orig_y_max = original_bbox
    orig_width = orig_x_max - orig_x_min
    orig_height = orig_y_max - orig_y_min

    # Center horizontally
    x_min = orig_x_min - (combined_width - orig_width) / 2
    x_max = orig_x_max + (combined_width - orig_width) / 2

    # Center vertically
    y_min = orig_y_min - (combined_height - orig_height) / 2
    y_max = orig_y_max + (combined_height - orig_height) / 2

    return int(x_min), int(x_max), int(y_min), int(y_max)


def process_combinations(list_obj_dirs, output_dir_path):
    num_of_obj = len(list_obj_dirs)
    os.makedirs(output_dir_path, exist_ok=True)

    # Find common angles for all objects
    common_angles = set(find_angles(list_obj_dirs[0]))
    for obj_dir in list_obj_dirs[1:]:
        common_angles = common_angles.intersection(find_angles(obj_dir))

    # Iterate over common angles and process images and masks
    for angle in common_angles:
        try:
            resized_images = []
            resized_masks = []
            bboxes = []
            images = []
            alphas = []

            # Find bounding boxes for all objects
            for obj_dir in list_obj_dirs:
                img_filename = get_angle_filenames(obj_dir, angle)
                img = imageio.imread(os.path.join(obj_dir, img_filename))
                images.append(img)
                alpha = img[:, :, 3]
                alphas.append(alpha)
                bbox = find_bounding_box(alpha)
                fixed_bbox = adjust_aspect_ratio(bbox, 1 if num_of_obj > 2 else 2)
                bboxes.append(fixed_bbox)

            # Find the largest bounding box
            max_width = max([bbox[1] - bbox[0] for bbox in bboxes])
            max_height = max([bbox[3] - bbox[2] for bbox in bboxes])

            # Center and resize all objects to match the largest bounding box
            for bbox, img, alpha in zip(bboxes, images, alphas):
                centered_bbox = center_within_combined_bbox(bbox, max_width, max_height)
                resized_image = crop_and_resize(img, centered_bbox,
                                                (256, 256) if num_of_obj > 2 else (256, 512))
                resized_mask = crop_and_resize(alpha, centered_bbox,
                                               (256, 256) if num_of_obj > 2 else (256, 512))
                resized_images.append(resized_image)
                resized_masks.append(resized_mask)

            angle_output_dir = os.path.join(output_dir_path, angle)
            concatenate_images_and_masks(resized_images, resized_masks, angle_output_dir, num_of_obj)
        except Exception as e:
            print(f"Error processing angle: {angle} with error: {e}")
