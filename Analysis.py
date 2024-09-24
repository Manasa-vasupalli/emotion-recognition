import os
import matplotlib.pyplot as plt
from PIL import Image

# Function to count images in folders, ignoring non-directory files like .DS_Store
def count_images_in_folders(data_dir):
    category_counts = {}
    
    categories = os.listdir(data_dir)
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        
        # Check if it's a directory, skip if it's not (e.g., .DS_Store)
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            category_counts[category] = count
    
    return category_counts

# Function to plot sample images from each category folder
def plot_sample_images(data_dir, num_samples=5):
    categories = os.listdir(data_dir)
    
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        
        # Skip non-directory files like .DS_Store
        if not os.path.isdir(folder_path):
            continue
        
        # Get sample images from the category folder
        sample_images = os.listdir(folder_path)[:num_samples]
        
        fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
        fig.suptitle(f"Emotion: {category}")
        
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        plt.show()


def plot_image_distribution(counts, title):
    categories = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(categories, values, color='Green')
    plt.title(title)
    plt.xlabel('Emotion Categories')
    plt.ylabel('Number of Images')
    plt.show()


# Directory paths
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Counting images in train and test folders
train_counts = count_images_in_folders(train_dir)
test_counts = count_images_in_folders(test_dir)


print("Train Set Distribution:\n", train_counts)
print("Test Set Distribution:\n", test_counts)

# Plot distribution of training and test data
plot_image_distribution(train_counts, 'Training Set Image Distribution')
plot_image_distribution(test_counts, 'Test Set Image Distribution')

# Plot sample images from the train directory
train_dir = 'Dataset/train'
plot_sample_images(train_dir)

