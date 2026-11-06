"""Fine-tune the optional Zero-DCE++ model using no-reference losses."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT))

from lumenx_core.models.zerodce import _network_class  # noqa: E402


def _images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _losses(torch: object, image: object, enhanced: object, curves: object) -> tuple[object, dict[str, float]]:
    import torch.nn.functional as functional

    gray_in = image.mean(dim=1, keepdim=True)
    gray_out = enhanced.mean(dim=1, keepdim=True)
    exposure = functional.avg_pool2d(gray_out, kernel_size=16, stride=16)
    exposure_loss = torch.mean(torch.abs(exposure - 0.60))

    patch_in = functional.avg_pool2d(gray_in, kernel_size=8, stride=8)
    patch_out = functional.avg_pool2d(gray_out, kernel_size=8, stride=8)
    input_dx = patch_in[:, :, :, 1:] - patch_in[:, :, :, :-1]
    input_dy = patch_in[:, :, 1:, :] - patch_in[:, :, :-1, :]
    output_dx = patch_out[:, :, :, 1:] - patch_out[:, :, :, :-1]
    output_dy = patch_out[:, :, 1:, :] - patch_out[:, :, :-1, :]
    spatial_loss = torch.mean((input_dx - output_dx) ** 2) + torch.mean((input_dy - output_dy) ** 2)

    means = enhanced.mean(dim=(2, 3))
    red, green, blue = means[:, 0], means[:, 1], means[:, 2]
    color_loss = torch.mean(torch.sqrt((red - green) ** 2 + (red - blue) ** 2 + (green - blue) ** 2 + 1e-8))

    guide = image.mean(dim=1, keepdim=True)
    grad_x = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])
    grad_y = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])
    curve_x = torch.abs(curves[:, :, :, 1:] - curves[:, :, :, :-1])
    curve_y = torch.abs(curves[:, :, 1:, :] - curves[:, :, :-1, :])
    smooth_x = torch.mean(curve_x * torch.exp(-10.0 * grad_x))
    smooth_y = torch.mean(curve_y * torch.exp(-10.0 * grad_y))
    smooth_loss = smooth_x + smooth_y

    total = 10.0 * exposure_loss + 1.0 * spatial_loss + 5.0 * color_loss + 200.0 * smooth_loss
    return total, {
        "exposure": float(exposure_loss.detach().item()),
        "spatial": float(spatial_loss.detach().item()),
        "color": float(color_loss.detach().item()),
        "smooth": float(smooth_loss.detach().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory of low-light training images")
    parser.add_argument("--output", type=Path, default=Path("zerodce_finetuned.pth"))
    parser.add_argument("--init", type=Path, help="Optional Zero-DCE++ checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=504)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.size % 12:
        parser.error("epochs and batch size must be positive; size must be at least 32 and divisible by 12")
    if not args.data.is_dir():
        parser.error(f"training image directory not found: {args.data}")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    files = _images(args.data)
    if not files:
        parser.error(f"no supported images found in {args.data}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    class LowLightImages(Dataset):
        def __len__(self) -> int:
            return len(files)

        def __getitem__(self, index: int) -> object:
            with Image.open(files[index]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = image.resize((args.size, args.size), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array.transpose(2, 0, 1).copy())

    loader = DataLoader(
        LowLightImages(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    model = _network_class()(scale_factor=12).to(device)
    if args.init:
        try:
            state = torch.load(args.init, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(args.init, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training {len(files)} unpaired images on {device}; output={args.output}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {"total": 0.0, "exposure": 0.0, "spatial": 0.0, "color": 0.0, "smooth": 0.0}
        steps = 0
        for batch in loader:
            image = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            enhanced, curves = model(image)
            total, components = _losses(torch, image, enhanced, curves)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["total"] += float(total.detach().item())
            for name, value in components.items():
                totals[name] += value
            steps += 1
        averages = {key: value / max(steps, 1) for key, value in totals.items()}
        print(f"epoch {epoch:03d}/{args.epochs}: " + " ".join(f"{key}={value:.5f}" for key, value in averages.items()))
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, args.output)
    print("Checkpoint saved. Review it on paired and unseen images before configuring it for runtime use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
