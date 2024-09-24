import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# This function is for counting the total no of images associated with a particular emotion
def count_images_in_folders(data_dir):
    categories = os.listdir(data_dir)
    category_counts = {}
    
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        count = len(os.listdir(folder_path))
        category_counts[category] = count
    
    return category_counts

# This function shows the distribution of each of the attributes
def plot_image_distribution(counts, title):
    categories = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(categories, values, color='Green')
    plt.title(title)
    plt.xlabel('Emotion Categories')
    plt.ylabel('Number of Images')
    plt.show()

# This function shows the mean and variance of the image categories 
def compute_mean_variance(folder):
    means = []
    variances = []
    
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        
        category_means = []
        category_variances = []
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).convert('L')  # Grayscale
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            
            category_means.append(np.mean(img_array))
            category_variances.append(np.var(img_array))
        
        means.append(np.mean(category_means))
        variances.append(np.mean(category_variances))
    
    return means, variances

#paths to your train and test directories
train_dir = 'Dataset/test'
test_dir = 'Dataset/train'

# Counting images in train and test folders
train_counts = count_images_in_folders(train_dir)
test_counts = count_images_in_folders(test_dir)


# Plot distribution of training and test data
plot_image_distribution(train_counts, 'Training Set Image Distribution')
plot_image_distribution(test_counts, 'Test Set Image Distribution')

train_means, train_variances = compute_mean_variance(train_dir)

# Printing mean and variance for each category
print("Train Set Mean Intensities:", train_means)
print("Train Set Variances:", train_variances)