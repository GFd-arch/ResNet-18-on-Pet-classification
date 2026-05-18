import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import swanlab

from dataset import OxfordPetsDataset
from model import ResNet18, ViTTiny
from trainer import Trainer, Evaluator


def main(model_type='resnet18', pretrained=True, epochs=300, batch_size=64,
         img_size=224, lr_backbone=2.5e-3, lr_fc=2.5e-3, weight_decay=2e-3):

    saved_path = "saved_models"
    os.makedirs(saved_path, exist_ok=True)

    experiment_name = f"{model_type}"

    if not pretrained:
        experiment_name += "_scratch"

    swanlab.init(
        project="pet-classification",
        experiment_name=experiment_name
    )

    # dataset
    ox = OxfordPetsDataset(img_size=img_size)

    train_dataset, test_dataset = ox.get_dataloaders(
        batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ViT-Tiny
    if model_type == 'vittiny':

        if pretrained is False:
            raise ValueError("ViT-Tiny only supports pretrained training.")

        model = ViTTiny(
            pretrained=True,
            out=37
        ).to(device)

        backbones, head = [], []

        for name, param in model.named_parameters():
            if 'head' in name:
                head.append(param)
            else:
                backbones.append(param)

        optimizer = optim.AdamW(
            [
                {'params': backbones, 'lr': lr_backbone},
                {'params': head, 'lr': lr_fc}
            ],
            weight_decay=weight_decay
        )

        def lr_lambda(epoch):
            warmup_epochs = 5

            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            return 0.5 * (
                1 + math.cos(
                    (epoch - warmup_epochs)
                    / (epochs - warmup_epochs)
                    * math.pi
                )
            )

        scheduler = LambdaLR(optimizer, lr_lambda)

    # ResNet18
    elif model_type == 'resnet18':

        model = ResNet18(
            pretrained=pretrained,
            out=37
        ).to(device)

        # from scratch
        if not pretrained:

            optimizer = optim.SGD(
                model.parameters(),
                lr=lr_backbone,
                momentum=0.9,
                weight_decay=weight_decay
            )

        # with pretrained
        else:
            backbones, fc = [], []

            for name, param in model.named_parameters():

                if 'fc' in name:
                    fc.append(param)
                else:
                    backbones.append(param)

            optimizer = optim.SGD(
                [
                    {'params': backbones, 'lr': lr_backbone},
                    {'params': fc, 'lr': lr_fc}
                ],
                momentum=0.9,
                weight_decay=weight_decay
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    else:
        raise ValueError("Invalid model type.")

    # train
    trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        criterion,
        optimizer,
        scheduler,
        device
    )

    trainer.train(epochs=epochs)

    # eval
    evaluator = Evaluator(model, test_dataset, device)

    acc = evaluator.evaluate()

    print(f"Accuracy: {acc:.4f}")

    swanlab.log({
        "eval/acc": acc
    })

    swanlab.finish()

    # save
    if model_type == 'resnet18':

        if pretrained:
            save_name = "resnet18_pretrained.pth"
        else:
            save_name = "resnet18_scratch.pth"

    else:
        save_name = "vittiny_pretrained.pth"

    torch.save(
        model.state_dict(),
        os.path.join(saved_path, save_name)
    )


# command line
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "vittiny"]
    )

    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train from scratch"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=2.5e-3
    )

    parser.add_argument(
        "--lr-fc",
        type=float,
        default=2.5e-3
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-3
    )

    args = parser.parse_args()

    # ViT not allowing scratch
    if args.model == "vittiny" and args.scratch:
        raise ValueError("ViT-Tiny does not support scratch training.")

    pretrained = not args.scratch

    main(
        model_type=args.model,
        pretrained=pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr_backbone=args.lr_backbone,
        lr_fc=args.lr_fc,
        weight_decay=args.weight_decay
    )