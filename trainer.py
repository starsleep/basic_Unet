import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.unet import Unet
from dataset.dataset import CustomDataSet

EVENTS_DIR = "./workspace/events"
CHECK_POINT_DIR = "./workspace/checkpoint"
SPLITES_DIR = "./splits"
DATASET_DIR = "../../../dataset"

TRAIN = "train"
VALIDATION = "validation"


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

class Trainer():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_ch = 3
        self.out_ch = 1
        self.epochs = 50
        self.batch_size = 4
        self.height = 512
        self.width = 512
        self.lr = 1e-4
        self.splits_file_path = SPLITES_DIR
        self.splits_train_name = "train_splits.txt"
        self.splits_val_name = "validation_splits.txt"
        self.dataset_dir = DATASET_DIR
        self.save_frequency = 10
        self.val_frequency = 1
        
        self.model = Unet(in_channels = self.in_ch, out_channels = self.out_ch)
        self.model.to(self.device)

        self.fn_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 1e-4)

        self.dataset = CustomDataSet(
            os.path.join(self.splits_file_path, self.splits_train_name),
            self.dataset_dir,
            self.height,
            self.width,
            True
        )
        self.dataloader = DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=True,
            num_workers = 0
        )

        self.val_dataset = CustomDataSet(
            os.path.join(self.splits_file_path, self.splits_val_name),
            self.dataset_dir,
            self.height,
            self.width,
            True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            1,
            shuffle=False,
            num_workers = 0
        )

        self.val_iter = iter(self.val_dataloader)

        self.num_total_steps = self.dataset.__len__() // self.batch_size * self.epochs

        self.writers = {}
        for mode in [TRAIN, VALIDATION]:
            self.writers[mode] = SummaryWriter(os.path.join(EVENTS_DIR, mode))


    def train(self):
        
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.epochs):
            self.process_epoch()

            if (self.epoch + 1) % self.save_frequency == 0:
                self.save_model()

    def process_epoch(self):
        self.model.train()

        for batch_idx, inputs in enumerate(self.dataloader):
            
            start = time.time()

            loss = self.process_batch(inputs, TRAIN)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            duration = time.time() - start
            self.step += 1

            self.log(TRAIN, loss.item())
            self.print_log(batch_idx, duration, loss)

            if (batch_idx % self.val_frequency) == 0:
                self.validation()
            

    def process_batch(self, inputs, mode)->torch.Tensor:
        
        (g_data, g_label) = inputs
        
        g_data = g_data.to(self.device)
        g_label = g_label.to(self.device)

        predic = self.model(g_data)
        
        if mode == TRAIN:
            loss = self.fn_loss(predic, g_label)
        elif mode == VALIDATION:
            loss = self.IoU(predic,g_label)

        return loss
        

    def validation(self):
        self.model.eval()

        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_dataloader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            loss = self.process_batch(inputs, VALIDATION)
            self.log(VALIDATION, loss.item())

        self.model.train()

    def log(self, mode, loss):
        writer = self.writers[mode]
        writer.add_scalar("{}".format(mode), loss, self.step)

    def print_log(self, batch_idx, duration, loss):
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = ( self.num_total_steps / self.step - 1.0 ) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        
        print(print_string.format(self.epoch, self.optim.state_dict()['param_groups'][0]['lr'],
                                    batch_idx, samples_per_sec, loss,sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def IoU(self, predic: torch.Tensor, label: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """객체 = 흰색(255) → 1
        배경 = 검정색(0) → 0
        """
        # ----- 1) Shape 정규화 -----
        if predic.dim() == 4:
            pred = predic.squeeze(1)  # (B,H,W)
            lab = label.squeeze(1)
        else:
            pred = predic
            lab = label

        pred = torch.sigmoid(pred)
        pred_bin = (pred > threshold).float()

        lab_bin = (lab > threshold).float()  # 흰색 픽셀만 객체로 판단

        intersection = (pred_bin * lab_bin).sum(dim=(-1, -2))
        union = pred_bin.sum(dim=(-1, -2)) + lab_bin.sum(dim=(-1, -2)) - intersection

        iou = intersection / (union + 1e-7)

        return iou
    

    def save_model(self ,save_onnx : bool = False):
        epoch_dir = os.path.join(CHECK_POINT_DIR, f"epoch_{self.epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)

        pth_path = os.path.join(epoch_dir, "unet.pth")
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            pth_path
        )

        if save_onnx:
            onnx_path = os.path.join(epoch_dir, "unet.onnx")

            was_training = self.model.training
            self.model.eval()

            dummy_input = torch.randn(1, self.in_ch, self.height, self.width, device=self.device)

            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            # 원래 모드로 복구
            if was_training:
                self.model.train()


   


