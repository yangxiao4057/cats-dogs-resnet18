"""训练、测试和预测共用的数据读取与模型构建。"""

import csv
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_rgb(path):
    # 与去重阶段一致：处理 EXIF 方向，并统一为三通道。
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class ManifestDataset(Dataset):
    def __init__(self, project, split):
        self.project = project
        with (project / "data" / "splits" / f"{split}.csv").open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"{split} 清单为空")
        self.transform = WEIGHTS.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_rgb(self.project / row["path"])
        return self.transform(image), int(row["label"])


def build_model(pretrained=True):
    model = resnet18(weights=WEIGHTS if pretrained else None)
    # 先冻结旧模型，再新建 fc；新建层默认允许求梯度。
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_training_mode(model):
    # eval 不会关闭自动求导；它让主干 BatchNorm 使用固定统计量。
    model.eval()
    model.fc.train()


def load_checkpoint(path, device):
    # checkpoint 含完整主干；重新加载不再下载 ImageNet 权重。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != {"cat": 0, "dog": 1}:
        raise ValueError("checkpoint 类别映射不符合预期")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
