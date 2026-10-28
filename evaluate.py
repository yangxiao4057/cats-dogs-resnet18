"""重新加载最佳权重，只评估测试集并保存图表；不更新模型。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from project_utils import ManifestDataset, load_checkpoint, load_rgb


def plot_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for true in range(2):
        for predicted in range(2):
            ax.text(predicted, true, str(matrix[true][predicted]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["cat", "dog"],
           yticklabels=["cat", "dog"], xlabel="Predicted label", ylabel="True label",
           title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cases(project, records, path, title):
    # 固定取清单中最先出现的案例，避免主观挑选；最多展示 12 张。
    records = records[:12]
    if not records:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cases", ha="center", va="center")
        ax.axis("off")
    else:
        columns = min(4, len(records))
        rows = (len(records) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.4), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, record in zip(axes.flat, records):
            ax.imshow(load_rgb(project / record["path"]))
            ax.set_title(f"{Path(record['path']).parent.name}/{Path(record['path']).name}\n"
                         f"true={record['true_class']} pred={record['predicted_class']}\n"
                         f"p(pred)={record['predicted_probability']:.3f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(device_name):
    project = Path(__file__).resolve().parent
    output = project / "outputs" / "test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("已有测试结果，请直接查看 outputs/test，避免覆盖。")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(project / "outputs/best.pt", device)
    # 检查训练、验证清单未被改动，再读取独立测试集。
    for name, expected in checkpoint["config"]["manifest_sha256"].items():
        actual = hashlib.sha256((project / f"data/splits/{name}.csv").read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} 清单与训练时不一致")
    data = ManifestDataset(project, "test")
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
    names = {index: name for name, index in checkpoint["class_to_idx"].items()}
    matrix = [[0, 0], [0, 0]]
    records, total_loss, offset = [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    print(f"重新加载最佳权重：epoch={checkpoint['epoch']}，device={device}", flush=True)
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images.to(device))
            total_loss += criterion(scores, labels.to(device)).item()
            probabilities = scores.softmax(dim=1).cpu()
            for label, probability in zip(labels.tolist(), probabilities.tolist()):
                predicted = max(range(2), key=lambda index: probability[index])
                matrix[label][predicted] += 1
                records.append({"path": data.rows[offset]["path"], "true_label": label,
                                "true_class": names[label], "predicted_label": predicted,
                                "predicted_class": names[predicted], "correct": predicted == label,
                                "cat_probability": probability[0], "dog_probability": probability[1],
                                "predicted_probability": probability[predicted]})
                offset += 1
    correct = [record for record in records if record["correct"]]
    errors = [record for record in records if not record["correct"]]
    metrics = {"checkpoint_epoch": checkpoint["epoch"], "device": str(device),
               "test_size": len(data), "correct": len(correct), "errors": len(errors),
               "accuracy": len(correct) / len(data), "loss": total_loss / len(data),
               "class_to_idx": checkpoint["class_to_idx"],
               "confusion_matrix_rows_true_columns_predicted": matrix,
               "test_manifest_sha256": hashlib.sha256((project / "data/splits/test.csv").read_bytes()).hexdigest(),
               "checkpoint_sha256": hashlib.sha256((project / "outputs/best.pt").read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    for filename, selected in (("predictions.csv", records), ("errors.csv", errors)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(selected)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    plot_matrix(matrix, output / "confusion_matrix.png")
    plot_cases(project, correct, output / "correct_examples.png", "Correct test predictions (first 12)")
    plot_cases(project, errors, output / "error_examples.png", "Incorrect test predictions (first 12)")
    print(f"测试准确率：{len(correct)}/{len(data)} = {metrics['accuracy']:.2%}", flush=True)
    print(f"混淆矩阵（行=真实，列=预测；顺序 cat, dog）：{matrix}", flush=True)
    print(f"错误数量：{len(errors)}；结果目录：{output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    main(parser.parse_args().device)
