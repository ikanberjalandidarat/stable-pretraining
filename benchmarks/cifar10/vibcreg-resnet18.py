"""VIbCReg training on CIFAR-10.

Follows the solo-learn recipe behind the reference result (ResNet-18,
1000 epochs, 91.18% online linear top-1): 2048-d GELU projector ending in
IterNorm, loss weights 25/25/200, LARS with lr 0.3, 10 warmup epochs.

Usage:
    python benchmarks/cifar10/vibcreg-resnet18.py
    python benchmarks/cifar10/vibcreg-resnet18.py --epochs 200 --wandb-project my-project
    python benchmarks/cifar10/vibcreg-resnet18.py --epochs 2 --limit-batches 5  # smoke check
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--wandb-project", type=str, default=None)
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument(
    "--limit-batches",
    type=int,
    default=None,
    help="Cap train/val batches per epoch, for a quick end-to-end check.",
)
args = parser.parse_args()


def _view_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


vibcreg_transform = transforms.MultiViewTransform(
    [_view_transform(), _view_transform()]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=vibcreg_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True,
    persistent_workers=args.num_workers > 0,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 2048),
    spt.backbone.IterNorm(2048, num_groups=64, T=5),
)

WARMUP_EPOCHS = 10


def _warmup_cosine(optimizer, module):
    # Warmup starts at 3e-5 (= 1e-4 * lr) as in solo-learn; total steps are
    # only known once the trainer is attached.
    total_steps = module.trainer.estimated_stepping_batches
    return spt.optim.LinearWarmupCosineAnnealing(
        optimizer,
        total_steps=total_steps,
        start_factor=1e-4,
        end_lr=0.0,
        peak_step=max(1, int(total_steps * WARMUP_EPOCHS / args.epochs)),
    )


module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.vibcreg,
    vibcreg_loss=spt.losses.VIbCRegLoss(
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=200.0,
    ),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": _warmup_cosine,
        "interval": "step",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

logger = True
if args.wandb_project:
    logger = WandbLogger(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"vibcreg-resnet18-{args.epochs}ep",
        log_model=False,
    )

trainer = pl.Trainer(
    max_epochs=args.epochs,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe],
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    logger=logger,
    limit_train_batches=args.limit_batches,
    limit_val_batches=args.limit_batches,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
