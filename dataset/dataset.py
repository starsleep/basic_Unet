import torch
import torch.utils.data as data

import os
import numpy as np

import cv2

def opencv_loader(path, channel : int = 3, isRGB : bool = True):

    data : np.ndarray = None

    if channel == 3:
        data = cv2.imread(path, cv2.IMREAD_COLOR)
    elif channel == 1:
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if isRGB and channel == 3:
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)

    return data

def ResizeTranform(x : np.ndarray, size:tuple[int , int])-> np.ndarray:
    return cv2.resize(x, size)

def NormTranform(x : np.ndarray)-> np.ndarray:
    return x.astype(np.float32) / 255.0

def InvertTranform(x : np.ndarray)-> np.ndarray:
    _, inv = cv2.threshold(x, 128, 255, cv2.THRESH_BINARY_INV)
    return inv

def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}")

    # 단일채널이면 (H,W,1) 형태로 확장
    if img.ndim == 2:
        img = img[..., None]  # (H,W,1)
    elif img.ndim != 3:
        raise ValueError(f"Invalid image shape {img.shape}, expected 2D or 3D")

    # NumPy → Tensor 변환
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
    return tensor

class CustomDataSet(data.Dataset):
    def __init__(self, splits_file_path, dataset_dir, height, width, is_train : bool):

        self.splits_file_path = splits_file_path
        self.dataset_dir = dataset_dir

        self.data_dir = os.path.join(dataset_dir, "data")
        self.label_dir = os.path.join(dataset_dir, "label")

        self.height = height
        self.width = width

        self.image_list = self._load_split_file()

        self.loader = opencv_loader

    def _load_split_file(self):
        image_names = []
        with open(self.splits_file_path, "r") as f:
            for line in f:
                name = line.strip()
                if name != "":
                    image_names.append(name)
        return image_names
    
    def __len__(self):
        return len(self.image_list)

    #객체는 흰색, 배경은 검은색
    def __getitem__(self, idx)->tuple[torch.Tensor , torch.Tensor]:
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]

        data_path = os.path.join(self.data_dir, img_name + ".bmp")
        label_path = os.path.join(self.label_dir, base_name + ".png")

        data = self.loader(data_path, channel=3, isRGB=True)
        label = self.loader(label_path, channel=1, isRGB=False)

        data, label = self.preprocess(data, label)

        t_data = np_to_tensor(data)
        t_label = np_to_tensor(label)

        return t_data, t_label

    def preprocess(self, data : np.ndarray, label : np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        data = ResizeTranform(data, (self.height,self.width))
        label = ResizeTranform(label, (self.height,self.width))

        data = NormTranform(data)

        label = InvertTranform(label)
        label = NormTranform(label)
        
        return data , label