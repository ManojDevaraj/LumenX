"""Paired and custom image metrics with explicit model availability."""

from __future__ import annotations

import math
import importlib.util

import cv2
import numpy as np

from .diagnose import luminance
from .guardrails import _chroma_delta_e


class MetricSuiteImpl:
    """CPU metrics for the classical engine; learned IQA stays unavailable."""

    def __init__(self, cfg: dict[str, dict[str, float | int]]):
        self._cfg = cfg
        self._iqa_models: dict[str, object] = {}
        self._iqa_errors: dict[str, str] = {}
        self._torch = None

    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
        pred, gt = self._pair(pred, gt)
        pred_f = pred.astype(np.float32)
        gt_f = gt.astype(np.float32)
        mse = float(np.mean((pred_f - gt_f) ** 2, dtype=np.float64))
        psnr = float(10.0 * math.log10((255.0**2) / mse)) if mse > 0.0 else None
        pred_mean = float(np.mean(pred_f))
        gt_mean = float(np.mean(gt_f))
        adjusted = np.clip(pred_f - pred_mean + gt_mean, 0.0, 255.0)
        adjusted_mse = float(np.mean((adjusted - gt_f) ** 2, dtype=np.float64))
        adjusted_psnr = (
            float(10.0 * math.log10((255.0**2) / adjusted_mse))
            if adjusted_mse > 0.0
            else None
        )
        ssim: float | None = None
        try:
            from skimage.metrics import structural_similarity

            min_side = min(pred.shape[:2])
            win_size = min(7, min_side if min_side % 2 else min_side - 1)
            if win_size >= 3:
                ssim = float(
                    structural_similarity(
                        gt,
                        pred,
                        channel_axis=2,
                        data_range=255,
                        win_size=win_size,
                    )
                )
        except ImportError:
            ssim = None
        except ValueError:
            ssim = None
        lpips = self._paired_iqa("lpips", pred, gt)
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "psnr_gt_mean": adjusted_psnr,
        }

    def unpaired(self, img: np.ndarray) -> dict[str, float | None]:
        self._rgb(img)
        return {
            name: self._single_iqa(name, img)
            for name in ("niqe", "brisque", "clipiqa", "musiq")
        }

    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]:
        inp, out = self._pair(inp, out)
        guardrail_cfg = self._cfg["guardrails"]
        diagnosis_cfg = self._cfg["diagnosis"]
        inp_f = inp.astype(np.float32) / 255.0
        out_f = out.astype(np.float32) / 255.0
        edge_in = self._edge_magnitude(luminance(inp_f))
        edge_out = self._edge_magnitude(luminance(out_f))
        supported = edge_in > max(
            float(np.percentile(edge_in, float(guardrail_cfg["reliable_edge_percentile"]))),
            float(guardrail_cfg["reliable_edge_floor"]),
        )
        edge_retention: float | None = None
        if int(np.count_nonzero(supported)) >= int(guardrail_cfg["minimum_edge_pixels"]):
            before = edge_in[supported].astype(np.float64)
            after = edge_out[supported].astype(np.float64)
            epsilon = float(guardrail_cfg["edge_std_epsilon"])
            if float(np.std(before)) > epsilon and float(np.std(after)) > epsilon:
                corr = float(np.corrcoef(before, after)[0, 1])
                if math.isfinite(corr):
                    edge_retention = float(np.clip(corr, -1.0, 1.0))
        in_noise = self._noise_sigma(luminance(inp_f))
        out_noise = self._noise_sigma(luminance(out_f))
        return {
            "clip_pct_in": float(
                np.mean(
                    np.max(inp_f, axis=2) > float(diagnosis_cfg["highlight_clip_threshold"]),
                    dtype=np.float64,
                )
                * 100.0
            ),
            "clip_pct_out": float(
                np.mean(
                    np.max(out_f, axis=2) > float(diagnosis_cfg["highlight_clip_threshold"]),
                    dtype=np.float64,
                )
                * 100.0
            ),
            "chroma_delta_e": _chroma_delta_e(
                inp_f,
                out_f,
                float(guardrail_cfg["chroma_mid_low"]),
                float(guardrail_cfg["chroma_mid_high"]),
                float(guardrail_cfg["chroma_max_span"]),
                int(guardrail_cfg["minimum_metric_pixels"]),
            ),
            "edge_retention": edge_retention,
            "noise_sigma_in": in_noise,
            "noise_sigma_out": out_noise,
        }

    def confidence_errors(
        self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray
    ) -> dict[str, float | None]:
        self._pair(inp, pred)
        self._pair(pred, gt)
        inp_f = inp.astype(np.float32) / 255.0
        pred_f = pred.astype(np.float32) / 255.0
        gt_f = gt.astype(np.float32) / 255.0
        gray = luminance(inp_f)
        mean = cv2.boxFilter(gray, -1, (15, 15), normalize=True)
        square_mean = cv2.boxFilter(gray * gray, -1, (15, 15), normalize=True)
        local_variance = np.maximum(square_mean - mean * mean, 0.0)
        snr_proxy = mean / np.maximum(np.sqrt(local_variance), 1.0 / 255.0)
        classes = np.ones(gray.shape, dtype=np.uint8)
        classes[snr_proxy < 1.5] = 2
        classes[snr_proxy >= 8.0] = 0
        error = np.mean(np.abs(pred_f - gt_f), axis=2)
        result: dict[str, float | None] = {}
        for code, name in ((0, "recovered"), (1, "estimated"), (2, "unrecoverable")):
            mask = classes == code
            result[f"err_{name}"] = (
                float(np.mean(error[mask], dtype=np.float64)) if np.any(mask) else None
            )
            result[f"{name}_pct"] = float(np.mean(mask, dtype=np.float64) * 100.0)
        return result

    def directions(self) -> dict[str, str]:
        return {
            "psnr": "higher",
            "ssim": "higher",
            "lpips": "lower",
            "psnr_gt_mean": "higher",
            "niqe": "lower",
            "brisque": "lower",
            "clipiqa": "higher",
            "musiq": "higher",
            "clip_pct_in": "lower",
            "clip_pct_out": "lower",
            "chroma_delta_e": "lower",
            "edge_retention": "higher",
            "noise_sigma_in": "lower",
            "noise_sigma_out": "lower",
            "err_recovered": "lower",
            "err_estimated": "lower",
            "err_unrecoverable": "lower",
        }

    def unavailable(self) -> dict[str, str]:
        unavailable = {
            name: self._iqa_errors.get(name, "The optional pyiqa model has not been loaded.")
            for name in ("lpips", "niqe", "brisque", "clipiqa", "musiq")
            if name not in self._iqa_models
        }
        return unavailable

    def health(self) -> dict[str, dict[str, object]]:
        installed = importlib.util.find_spec("pyiqa") is not None
        return {
            name: {
                "available": name in self._iqa_models,
                "installed": installed,
                "message": self._iqa_errors.get(name),
            }
            for name in ("lpips", "niqe", "brisque", "clipiqa", "musiq")
        }

    def _model(self, name: str) -> object | None:
        if name in self._iqa_models:
            return self._iqa_models[name]
        if name in self._iqa_errors:
            return None
        if importlib.util.find_spec("pyiqa") is None:
            self._iqa_errors[name] = "Install lumenx-core[models] to load pyiqa metrics."
            return None
        try:
            import pyiqa
            import torch

            model = pyiqa.create_metric(name, device="cpu", as_loss=False)
            model.eval()
            self._iqa_models[name] = model
            self._torch = torch
            return model
        except Exception as exc:
            self._iqa_errors[name] = f"Could not load {name}: {type(exc).__name__}: {exc}"
            return None

    def _tensor(self, image: np.ndarray) -> object:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch.from_numpy(
            image.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0)

    def _single_iqa(self, name: str, image: np.ndarray) -> float | None:
        model = self._model(name)
        if model is None:
            return None
        try:
            with self._torch.inference_mode():
                value = model(self._tensor(image))
            score = float(value.detach().float().mean().cpu().item())
            if not math.isfinite(score):
                raise ValueError("metric returned a non-finite score")
            return score
        except Exception as exc:
            self._iqa_errors[name] = f"Could not evaluate {name}: {type(exc).__name__}: {exc}"
            self._iqa_models.pop(name, None)
            return None

    def _paired_iqa(self, name: str, pred: np.ndarray, gt: np.ndarray) -> float | None:
        model = self._model(name)
        if model is None:
            return None
        try:
            with self._torch.inference_mode():
                value = model(self._tensor(pred), self._tensor(gt))
            score = float(value.detach().float().mean().cpu().item())
            if not math.isfinite(score):
                raise ValueError("metric returned a non-finite score")
            return score
        except Exception as exc:
            self._iqa_errors[name] = f"Could not evaluate {name}: {type(exc).__name__}: {exc}"
            self._iqa_models.pop(name, None)
            return None

    @staticmethod
    def _rgb(img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError("metrics expect uint8 RGB arrays")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("metrics expect HxWx3 RGB arrays")

    @classmethod
    def _pair(cls, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cls._rgb(a)
        cls._rgb(b)
        if a.shape != b.shape:
            raise ValueError("paired metrics expect arrays with identical shapes")
        return a, b

    @staticmethod
    def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(dx, dy)

    @staticmethod
    def _noise_sigma(gray: np.ndarray) -> float:
        residual = gray - cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(residual))
        return float(np.median(np.abs(residual - median)) * 1.4826)
