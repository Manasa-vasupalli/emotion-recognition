import os

# This function is for counting the total no of images associated with a particular emotion
def count_images_in_folders(data_dir):
    categories = os.listdir(data_dir)
    category_counts = {}
    
    for category in categories:
        folder_path = os.path.join(data_dir, category)
        count = len(os.listdir(folder_path))
        category_counts[category] = count
    
    return category_counts

#paths to your train and test directories
train_dir = 'Dataset/test'
test_dir = 'Dataset/train'

# Counting images in train and test folders
train_counts = count_images_in_folders(train_dir)
test_counts = count_images_in_folders(test_dir)

print("Train Set Distribution:\n")
for i, j in train_counts.items():
    print(i, j)

print("\nTest Set Distribution:\n")
for i, j in train_counts.items():
    print(i, j)