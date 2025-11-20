# crack classifier yes/no using resnet

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import csv
from torchvision.models import ResNet18_Weights

class CrackClassifier:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path) if model_path else self._initialize_model()
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.505, 0.494, 0.474], std=[0.098, 0.099, 0.099])
        ])

    def _load_model(self, model_path):
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _initialize_model(self):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
        model.to(self.device)
        model.eval()
        return model
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.005, save_every=10):
        # Data is unbalanced, we want to avoid false negatives more than false positives, there are 5 times more false labels
        class_weights = torch.tensor([1.0, 5.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                print(f'Epoch {epoch+1}/{epochs}, batch {i+1}/{len(train_loader)}', end='\r')
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            val_acc, val_re, val_pr = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Val Re: {val_re:.4f}, Val Pr: {val_pr:.4f}')
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            
            if (epoch + 1) % save_every == 0:
                save_path = f'./saved-model/crack_classifier_resnet18_epoch{epoch+1}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved to {save_path}')
            
    def evaluate(self, data_loader):
        self.model.eval()
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        confusion_matrix = [[tn, fp], [fn, tp]]
        
        return accuracy, recall, precision, confusion_matrix
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.item()  # 0 or 1