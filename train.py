"""仅训练 ResNet18 分类层，按验证准确率保存最佳模型；不读取测试集。"""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存图片，不依赖桌面窗口。
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from project_utils import ManifestDataset, build_model, set_seed, set_training_mode


def run_epoch(model, loader, criterion, device, optimizer=None, deadline=None):
    training = optimizer is not None
    if training:
        set_training_mode(model)
    else:
        model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            scores = model(images)  # 两个原始分数，交叉熵前不加 Softmax。
            loss = criterion(scores, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += batch_size
    return {"loss": loss_sum / count, "acc": correct / count}


def save_history(output, history):
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric, title in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker="o", label=split)
        axis.set(xlabel="Epoch", ylabel=title, xticks=epochs)
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def main(batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请在本机 WSL 的 d2l 环境运行，不自动改用 CPU 正式训练。")
    project = Path(__file__).resolve().parent
    output = project / "outputs"
    # 避免误运行覆盖已有实验。
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("outputs 已有结果。请先检查并备份，避免覆盖已有实验。")
    set_seed(42)
    device = torch.device("cuda:0")
    class_to_idx = json.loads((project / "data/splits/class_to_idx.json").read_text())
    if class_to_idx != {"cat": 0, "dog": 1}:
        raise ValueError("类别映射与预期不符")
    train_data = ManifestDataset(project, "train")
    val_data = ManifestDataset(project, "val")
    generator = torch.Generator().manual_seed(42)
    # num_workers=0 避免初次实践中的多进程问题；训练集打乱，验证集不打乱。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print("加载官方 ImageNet 预训练权重（首次运行可能需要下载约 45MB）…", flush=True)
    model = build_model(pretrained=True).to(device)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable != ["fc.weight", "fc.bias"]:
        raise RuntimeError(f"可训练参数异常：{trainable}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    config = {
        "architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1",
        "seed": 42, "epochs": 5, "batch_size": batch_size, "learning_rate": 0.001,
        "optimizer": "Adam", "loss": "CrossEntropyLoss", "time_budget_seconds": 1200,
        "train_size": len(train_data), "val_size": len(val_data),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_batchnorm": "eval; running statistics fixed",
        "preprocessing": "EXIF transpose, RGB, IMAGENET1K_V1.transforms()",
        "class_to_idx": class_to_idx,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "gpu": torch.cuda.get_device_name(0),
        "manifest_sha256": {name: hashlib.sha256((project / f"data/splits/{name}.csv").read_bytes()).hexdigest()
                            for name in ("train", "val")},
    }
    output.mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    # 保存主干参数及 BN buffer 的副本，训练后核对是否真的保持不变。
    backbone_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                       if not name.startswith("fc.")}
    print(f"GPU: {config['gpu']} | train={len(train_data)}, val={len(val_data)}", flush=True)
    print(f"仅训练 {trainable}，共 {config['trainable_parameters']} 个参数", flush=True)
    history, best_acc, best_epoch = [], -1.0, None
    started = time.perf_counter()
    deadline = started + config["time_budget_seconds"]
    stop_reason = "completed_5_epochs"
    for epoch in range(1, 6):
        epoch_start = time.perf_counter()
        train = run_epoch(model, train_loader, criterion, device, optimizer, deadline)
        val = None if train is None else run_epoch(model, val_loader, criterion, device, deadline=deadline)
        if train is None or val is None:
            stop_reason = "time_budget_reached; partial_epoch_not_saved"
            print("达到20分钟预算，停止训练；未完成的轮次不用于模型选择。", flush=True)
            break
        row = {"epoch": epoch, "train_loss": train["loss"], "train_acc": train["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"],
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if val["acc"] > best_acc:  # 相同时保留较早轮次。
            best_acc, best_epoch = val["acc"], epoch
            torch.save({"model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        "class_to_idx": class_to_idx, "config": config,
                        "epoch": epoch, "val_acc": best_acc}, output / "best.pt")
        save_history(output, history)
        print(f"Epoch {epoch}/5 | train loss={train['loss']:.4f}, acc={train['acc']:.2%} | "
              f"val loss={val['loss']:.4f}, acc={val['acc']:.2%} | {row['seconds']:.1f}s", flush=True)
    unchanged = all(torch.equal(value, model.state_dict()[name].detach().cpu())
                    for name, value in backbone_before.items())
    summary = {"completed_epochs": len(history), "best_epoch": best_epoch,
               "best_val_acc": None if best_epoch is None else best_acc,
               "elapsed_seconds": time.perf_counter() - started, "stop_reason": stop_reason,
               "backbone_parameters_and_buffers_unchanged": unchanged,
               "test_evaluated": False}
    (output / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not unchanged:
        raise RuntimeError("主干参数或统计量发生变化，请勿使用本次结果。")
    if not history:
        raise RuntimeError("未完成任何完整轮次，没有最佳模型；请反馈日志。")
    print(f"训练结束。最佳轮次={best_epoch}，验证准确率={best_acc:.2%}；主干保持不变。", flush=True)
    print(f"结果目录：{output}。尚未读取测试集。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(16, 32), default=32)
    args = parser.parse_args()
    main(args.batch_size)
