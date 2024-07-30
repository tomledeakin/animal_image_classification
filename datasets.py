from torch.utils.data import Dataset, DataLoader
import os
import cv2
from PIL import Image
import pickle
from torchvision.transforms import Compose, ToTensor, Resize

class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]

        self.all_image_paths = []
        self.all_labels = []
        for index, category in enumerate(categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_image_paths[item]
        # image = cv2.imread(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    dataset = AnimalDataset(root="data/animals", is_train=True, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )

    for images, labels in dataloader:
        print(images.shape)
        print(labels)
        print("------------")