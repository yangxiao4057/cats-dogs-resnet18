"""加载最佳模型并预测一张图片。"""

import argparse
from pathlib import Path

import torch

from project_utils import WEIGHTS, load_checkpoint, load_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="图片路径，含空格时请加引号")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    project = Path(__file__).resolve().parent
    model, checkpoint = load_checkpoint(project / "outputs/best.pt", device)
    image = WEIGHTS.transforms()(load_rgb(args.image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0].cpu()
    names = {index: name for name, index in checkpoint["class_to_idx"].items()}
    predicted = int(probabilities.argmax())
    print(f"图片：{args.image}")
    print(f"预测：{names[predicted]}，模型概率：{probabilities[predicted].item():.2%}")
    for index in range(2):
        print(f"  {names[index]}: {probabilities[index].item():.2%}")
    print("模型概率未经校准，不代表真实可靠率。")


if __name__ == "__main__":
    main()
