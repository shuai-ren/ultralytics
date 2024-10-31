import os
import shutil
import random

def create_validation_set(image_dir, label_dir, val_ratio):
    train_image_dir = os.path.join(image_dir, 'train')
    train_label_dir = os.path.join(label_dir, 'train')
    val_image_dir = os.path.join(image_dir, 'val')
    val_label_dir = os.path.join(label_dir, 'val')
    
    # Create val directories if they don't exist
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Get list of jpg files in train image directory
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith('.jpg')]
    
    # Select files for validation set
    num_val = int(len(image_files) * val_ratio)
    val_files = random.sample(image_files, num_val)
    
    for image_file in val_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        
        image_path = os.path.join(train_image_dir, image_file)
        label_path = os.path.join(train_label_dir, label_file)
        
        # Check if the corresponding txt file exists
        if os.path.exists(label_path):
            # Move image and label files to validation directory
            shutil.move(image_path, os.path.join(val_image_dir, image_file))
            shutil.move(label_path, os.path.join(val_label_dir, label_file))
        else:
            print(f"Warning: Corresponding label file for {image_file} not found.")

# Example usage
image_dir = 'images'
label_dir = 'labels'
val_ratio = 0.2  # 20% of the data will be used for validation

create_validation_set(image_dir, label_dir, val_ratio)

