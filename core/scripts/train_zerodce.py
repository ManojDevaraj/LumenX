"""Fine-tune Zero-DCE++ with no-reference losses on an approved training split."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps

CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT))

from lumenx_core.models.zerodce_arch.model import enhance_net_nopool  # noqa: E402


def images(root: Path) -> list[Path]:
    """Find supported image files below the explicitly supplied training directory."""
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def losses(torch, image, enhanced, curves):
    """Compute exposure, spatial consistency, color constancy and curve smoothness."""
    import torch.nn.functional as functional

    gray_in, gray_out = image.mean(1, keepdim=True), enhanced.mean(1, keepdim=True)
    exposure = torch.mean(torch.abs(functional.avg_pool2d(gray_out, 16, 16) - 0.60))
    patch_in, patch_out = functional.avg_pool2d(gray_in, 8, 8), functional.avg_pool2d(gray_out, 8, 8)
    spatial = (torch.mean(((patch_in[:, :, :, 1:] - patch_in[:, :, :, :-1]) -
                           (patch_out[:, :, :, 1:] - patch_out[:, :, :, :-1])) ** 2)
               + torch.mean(((patch_in[:, :, 1:, :] - patch_in[:, :, :-1, :]) -
                             (patch_out[:, :, 1:, :] - patch_out[:, :, :-1, :])) ** 2))
    means = enhanced.mean(dim=(2, 3))
    red, green, blue = means[:, 0], means[:, 1], means[:, 2]
    color = torch.mean(torch.sqrt((red-green)**2 + (red-blue)**2 + (green-blue)**2 + 1e-8))
    guide = image.mean(1, keepdim=True)
    smooth = torch.mean(torch.abs(curves[:, :, :, 1:] - curves[:, :, :, :-1])
                        * torch.exp(-10 * torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])))
    smooth += torch.mean(torch.abs(curves[:, :, 1:, :] - curves[:, :, :-1, :])
                        * torch.exp(-10 * torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])))
    total = 10 * exposure + spatial + 5 * color + 200 * smooth
    return total, {"total": total.detach().item(), "exposure": exposure.detach().item(),
                   "spatial": spatial.detach().item(), "color": color.detach().item(),
                   "smooth": smooth.detach().item()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=CORE_ROOT / "weights/zerodce_pp_finetuned.pth")
    parser.add_argument("--init", type=Path, default=CORE_ROOT / "weights/zerodce_pp.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=504)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()
    if not data.is_dir() or args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.size % 12:
        parser.error("Training directory must exist; epochs/batch positive; size >=32 and divisible by 12")
    files = images(data)
    if not files:
        parser.error(f"No images found in {data}")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Fine-tuning requires CUDA; no CPU fallback is allowed")
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    class LowImages(Dataset):
        def __len__(self): return len(files)
        def __getitem__(self, index):
            with Image.open(files[index]) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
                array = np.asarray(rgb, dtype=np.float32) / 255
            return torch.from_numpy(array.transpose(2, 0, 1).copy())

    loader = DataLoader(LowImages(), batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=False)
    model = enhance_net_nopool(scale_factor=12).to(args.device)
    state = torch.load(args.init, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"data={data}\nimages={len(files)}\ndevice={args.device}\ninit={args.init.resolve()}\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = {key: 0.0 for key in ("total", "exposure", "spatial", "color", "smooth")}
            steps = 0
            for batch in loader:
                image = batch.to(args.device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                enhanced, curves = model(image)
                total, values = losses(torch, image, enhanced, curves)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                for key, value in values.items():
                    sums[key] += value
                steps += 1
            report = " ".join(f"{key}={value / steps:.6f}" for key, value in sums.items())
            line = f"epoch={epoch}/{args.epochs} {report}"
            print(line, flush=True)
            stream.write(line + "\n")
            stream.flush()
            temporary = output.with_suffix(output.suffix + ".part")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"checkpoint={output}\nsha256={digest}\nlog={log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

