import torch
from data.loaders import get_dataloader
from models.chronoforget import ChronoForget
from models.forget_mi import ForgetMI
from trainers.trainer import ChronoForgetTrainer

DATASETS = ['MIMIC-III', 'NIH ChestX-ray14', 'ADNI', 'ISIC Skin', 'eICU']

def run_chronoforget_on_all_datasets(data_root='/home/phd/datasets/'):
    results = {}

    for dataset_name in DATASETS:
        print(f"\n🔄 Processing dataset: {dataset_name}")

        if dataset_name == 'MIMIC-III':
            dl_forget = get_dataloader(dataset_name, root_dir=os.path.join(data_root, 'mimic'), batch_size=32)
            dl_retain = get_dataloader(dataset_name, root_dir=os.path.join(data_root, 'mimic'), batch_size=32)

        elif dataset_name == 'NIH ChestX-ray14':
            dl_forget = get_dataloader(
                dataset_name,
                img_dir=os.path.join(data_root, 'nih/images'),
                csv_file=os.path.join(data_root, 'nih/labels.csv'),
                batch_size=32
            )
            dl_retain = dl_forget

        elif dataset_name == 'ADNI':
            dl_forget = get_dataloader(dataset_name, root_dir=os.path.join(data_root, 'adni'), batch_size=8)
            dl_retain = dl_forget

        elif dataset_name == 'ISIC Skin':
            dl_forget = get_dataloader(
                dataset_name,
                img_dir=os.path.join(data_root, 'isic/images'),
                csv_file=os.path.join(data_root, 'isic/labels.csv'),
                batch_size=32
            )
            dl_retain = dl_forget

        elif dataset_name == 'eICU':
            dl_forget = get_dataloader(dataset_name, root_dir=os.path.join(data_root, 'eicu'), batch_size=16)
            dl_retain = dl_forget

        # Initialize model
        base_model = ChronoForgetTrainer(BaseResNet(num_classes=2))
        chrono_forget_trainer = ChronoForget(base_model)

        # Run unlearning
        chrono_forget_trainer.unlearn(dl_forget, dl_retain)

        results[dataset_name] = {
            'chrono_forget': chrono_forget_trainer.model
        }

    return results

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = run_chronoforget_on_all_datasets()
    print("\n📊 Performance Summary:")
    for ds, model_dict in results.items():
        print(f"{ds}: Trained")