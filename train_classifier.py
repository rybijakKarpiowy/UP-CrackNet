from classifier import CrackClassifier
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    data_dir = './crack_segmentation_dataset/train/images/'
    model_save_path = './saved-model/crack_classifier_resnet18.pth'
    
    train_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.505, 0.494, 0.474], std=[0.098, 0.099, 0.099])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.505, 0.494, 0.474], std=[0.098, 0.099, 0.099])
    ])
    
    all_image_paths = []
    all_labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            all_image_paths.append(os.path.join(data_dir, filename))
            label = 0 if filename.startswith('noncrack') else 1
            all_labels.append(label)
            
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42)
    
    train_dataset = ClassifierDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = ClassifierDataset(val_paths, val_labels, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    classifier = CrackClassifier(model_path=None)
    classifier.train(train_loader, val_loader, epochs=25, lr=0.001, save_every=2)

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(classifier.model.state_dict(), model_save_path)
    
    # Evaluate on validation set
    val_accuracy, val_re, val_pr = classifier.evaluate(val_loader)
    print(f'Validation Accuracy: {val_accuracy:.4f}, Recall: {val_re:.4f}, Precision: {val_pr:.4f}')
    
    
if __name__ == '__main__':
    main()
    
    test_data = ClassifierDataset(
        image_paths=[os.path.join('./crack_segmentation_dataset/test/images/', f) for f in os.listdir('./crack_segmentation_dataset/test/images/')],
        labels=[0 if f.startswith('noncrack') else 1 for f in os.listdir('./crack_segmentation_dataset/test/images/')],
        transform=transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
        ])
    )
    
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)
    
    classifier = CrackClassifier(model_path='./saved-model/crack_classifier_resnet18.pth')
    
    test_accuracy, test_re, test_pr = classifier.evaluate(test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}, Recall: {test_re:.4f}, Precision: {test_pr:.4f}')
    