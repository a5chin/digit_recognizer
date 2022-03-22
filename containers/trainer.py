import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import build_dataloader
from .model import *
from .scheduler import *


class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.total_epoch = cfg.SETTINGS.TOTAL_EPOCH
        self.batch_size = cfg.DATA.BATCH_SIZE
        self.best_acc = 0.0
        self.train_dataloader, self.val_dataloader = build_dataloader(
            root=cfg.DATA.ROOT,
            ratio=cfg.DATA.RATIO,
            batch_size=self.batch_size
        )
        self.ckpt_path = Path(f"{cfg.MODEL.CKPT}")
        self.ckpt_path.mkdir(exist_ok=True)
        self.model = eval(f"{cfg.MODEL.NAME}")(num_classes=cfg.MODEL.NUM_CLASSES)
        self.model.to(self.device)

    def train(self):
        self.criterion = eval(f"nn.{self.cfg.SETTINGS.CRITERION}")()
        self.optimizer = eval(f"optim.{self.cfg.SETTINGS.OPTIMIZER}")(
            self.model.parameters(),
            lr=5e-3
        )
        self.scheduler = eval(self.cfg.SCHEDULER)

        for epoch in range(self.total_epoch):
            self.model.train()
            accuracies, losses = 0.0, 0.0

            with tqdm(self.train_dataloader) as pbar:
                pbar.set_description(f'[Epoch {epoch + 1}/{self.total_epoch}]')

                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    losses += loss.item()

                    loss.backward()
                    self.optimizer.step()

                    preds = outputs.argmax(axis=1)
                    accuracy = torch.sum(preds == labels) / self.batch_size
                    accuracies += accuracy.item()

                    pbar.set_postfix(OrderedDict(
                        Loss=loss.item(), Accuracy=accuracy.item()
                    ))

            self._write_summary(
                mode="train",
                epoch=epoch,
                accuracy=accuracies / len(self.train_dataloader),
                loss=losses / len(self.train_dataloader),
                lr=self.scheduler.get_last_lr()[0]
            )

            self.evaluate(
                model=self.model,
                dataloader=self.val_dataloader,
                epoch=epoch
            )

            self.scheduler.step()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader]=None,
        epoch: Optional[int]=None
    ):
        model.eval()
        accuracies = 0.0

        with torch.inference_mode():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                preds = outputs.argmax(axis=1)
                accuracies += torch.sum(preds == labels).item() / self.batch_size

        accuracy = accuracies / len(dataloader)
        self._write_summary(mode="val", epoch=epoch, accuracy=accuracy)

        if self.best_acc <= accuracy:
            self.best_acc = accuracy
            torch.save(self.model.state_dict(), "ckpt/best_ckpt.pth")


    def _write_summary(
            self,
            mode: str="train",
            epoch: int=0,
            **kwargs
        ) -> None:
            if epoch == 0:
                exec(f"self.{mode}_writer = SummaryWriter(log_dir='log/{mode}')")

            for key in kwargs:
                exec(f"self.{mode}_writer.add_scalar('{key}', {kwargs[key]}, {epoch})")
