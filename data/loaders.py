import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
import pandas as pd
import numpy as np

class ClinicalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class MIMICIIIDataset(ClinicalDataset):
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, transform)
        # Simulated or load real tabular data here
        self.tabular_data = np.random.rand(1000, 50)
        self.text_labels = np.random.randint(0, 2, size=1000)
        self.patient_ids = [f"p_{i}" for i in range(1000)]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        sample = {
            'tabular': torch.tensor(self.tabular_data[idx], dtype=torch.float32),
            'label': torch.tensor(self.text_labels[idx], dtype=torch.long),
            'patient_id': self.patient_ids[idx]
        }
        return sample


class NIHChestXray14Dataset(ClinicalDataset):
    def __init__(self, img_dir, csv_file, transform=None):
        super().__init__(img_dir, transform)
        self.df = pd.read_csv(csv_file)
        self.img_paths = self.df['Image Index']
        self.labels = self.df.drop(columns=['Image Index']).values

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32)
        }


class ADNIDataset(ClinicalDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.mri_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.nii.gz')]
        self.labels = np.random.randint(0, 2, size=len(self.mri_files))

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        mri = torch.rand(1, 64, 64, 64)  # Placeholder for MRI volume
        label = self.labels[idx]
        return {
            'mri': mri,
            'label': torch.tensor(label, dtype=torch.long)
        }


class ISICSkinLesionDataset(ClinicalDataset):
    def __init__(self, img_dir, csv_file, transform=None):
        super().__init__(img_dir, transform)
        self.df = pd.read_csv(csv_file)
        self.img_paths = self.df['image_path'].values
        self.labels = self.df['melanoma'].values

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.img_paths[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


class eICUDataset(ClinicalDataset):
    def __init__(self, root_dir, time_steps=5, transform=None):
        super().__init__(root_dir, transform)
        self.time_steps = time_steps
        self.vitals = np.random.rand(1000, time_steps, 12)
        self.notes = ["clinical note"] * 1000
        self.labels = np.random.randint(0, 2, size=1000)

    def __len__(self):
        return len(self.vitals)

    def __getitem__(self, idx):
        vitals = self.vitals[idx]
        note = self.notes[idx]
        label = self.labels[idx]

        return {
            'vitals': torch.tensor(vitals, dtype=torch.float32),
            'note': note,
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_dataloader(dataset_name, batch_size=32, shuffle=True, **kwargs):
    dataset_map = {
        "MIMIC-III": MIMICIIIDataset,
        "NIH ChestX-ray14": NIHChestXray14Dataset,
        "ADNI": ADNIDataset,
        "ISIC Skin": ISICSkinLesionDataset,
        "eICU": eICUDataset
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_class = dataset_map[dataset_name]
    dataset = dataset_class(**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)