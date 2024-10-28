import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from transformers import Trainer, TrainingArguments
import evaluate
import wandb
import os
from PIL import Image
import splitfolders
import numpy as np

# Inisialisasi Weights & Biases
run = wandb.init(
    project='trash-classification',
    config={
        'learning_rate': 1e-4,
        'loss': 'categorical_crossentropy',
        'epoch': 10,
        'batch_size': 32,
        'num_classes': 6
    }
)

wandb_config = wandb.config

# Direktori data
data_dir = 'data'
split_dir = '/tmp/data_split'

# Memisahkan dataset
splitfolders.ratio(data_dir, output=split_dir, seed=507, ratio=(.8, .2), group_prefix=None)

# Kelas dataset kustom
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform 
        self.img_labels = []
        self.img_paths = []

        for label in os.listdir(img_dir): 
            class_dir = os.path.join(img_dir, label)

            if os.path.isdir(class_dir):
                class_images = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
                self.img_paths.extend(class_images)
                self.img_labels.extend([os.listdir(img_dir).index(label)] * len(class_images))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB') 
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image) 

        return {
            'pixel_values': image, 
            'label_ids': torch.tensor(label),
            'labels': label
        }
    
# Transformasi untuk dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Membuat dataset pelatihan dan pengujian
train_dataset = ImageDataset(os.path.join(split_dir, 'train'), transform=train_transform)
test_dataset = ImageDataset(os.path.join(split_dir, 'val'), transform=test_transform)

# Model kustom
class CustomModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values)
        
        # If labels are provided, you can calculate the loss here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
            loss = loss_fct(outputs, labels)
            return {'loss': loss, 'logits': outputs}
        
        return outputs

# Fungsi untuk menghitung akurasi
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # Memuat model ResNet18
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, wandb_config.num_classes)
    model = CustomModel(base_model)

    # Mengatur argumen pelatihan
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=wandb_config.learning_rate,
        per_device_train_batch_size=wandb_config.batch_size,
        per_device_eval_batch_size=wandb_config.batch_size,  # Perbaiki 'bach_size' ke 'batch_size'
        num_train_epochs=wandb_config.epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='wandb',
        run_name='ResNet'
    )

    # Menginisialisasi Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Melatih model
    trainer.train()

    # Menyimpan model
    trainer.save_model('./results')  # Perbaiki path
    wandb.save('./results/*')
    wandb.finish()
