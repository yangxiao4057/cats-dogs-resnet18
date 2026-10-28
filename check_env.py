"""检查项目依赖，并实际执行一次 CUDA 运算；不下载权重或训练模型。"""

import platform
import sys

import matplotlib
import PIL
import torch
import torchvision
from torchvision.models import ResNet18_Weights


def main():
    print(f"操作系统: {platform.platform()}")
    print(f"Python 路径: {sys.executable}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"Pillow: {PIL.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"PyTorch 编译使用的 CUDA: {torch.version.cuda}")
    print("\n官方权重预处理（这里只查看配置，不下载权重）:")
    print(ResNet18_Weights.IMAGENET1K_V1.transforms())

    if not torch.cuda.is_available():
        print("\nCUDA 不可用。本次检查未通过；请在本机 WSL 的 d2l 环境运行并反馈输出。")
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {properties.name}")
    print(f"计算能力: {properties.major}.{properties.minor}")
    print(f"显存总量: {properties.total_memory / 1024**3:.2f} GiB")
    print(f"PyTorch 支持的 CUDA 架构: {torch.cuda.get_arch_list()}")
    try:
        # 不只查询显卡：真正启动 CUDA 运算，再与 CPU 结果比较。
        generator = torch.Generator().manual_seed(42)
        a = torch.randn(128, 128, generator=generator)
        b = torch.randn(128, 128, generator=generator)
        actual = a.to(device) @ b.to(device)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(actual.cpu(), a @ b, rtol=1e-3, atol=1e-3)
        # ResNet18 需要卷积：额外验证一次 cuDNN 前向和反向。
        conv = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
        output = conv(torch.ones(2, 3, 32, 32, device=device))
        output.square().mean().backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(output).all().item():
            raise RuntimeError("卷积输出包含非有限值")
    except (RuntimeError, AssertionError) as error:
        print(f"\nGPU 实际计算失败: {type(error).__name__}: {error}")
        return 1

    print("\n检查通过：依赖可导入，GPU 矩阵运算及卷积前向、反向均成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
