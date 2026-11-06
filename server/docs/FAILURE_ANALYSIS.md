# Failure analysis and review checklist

The application reports measurements and fallbacks so weak cases can be reviewed rather than hidden. `server/eval/mine_failures.py` ranks evaluator errors, unresolved guardrails, and confidence-class errors for manual inspection.

## Cases to include

- Near-black frames with little recoverable signal. Check the recoverability confidence, noise amplification, and whether the safe path prevents exaggerated colour.
- Small bright sources against dark backgrounds. Inspect clipping, halos, lamp colour, and the protected highlight region.
- Strongly nonuniform illumination. Inspect the illumination and fusion maps for bands or broad halos.
- High sensor noise, compression blocks, and chroma speckle. Compare edge retention with measured noise reduction.
- Coloured lighting and scenes with intentional casts. Verify bounded colour correction does not neutralize the scene.
- Fine texture, text, foliage, and repeating patterns. Check detail loss after guided denoising and proxy candidate selection.
- Unseen camera sources and resolutions. Record input dimensions, downscaling, device, model availability, and checkpoint hash.

## Interpreting the audit

Guardrail thresholds are configurable heuristics, not proof of perceptual quality. A flagged check may be a legitimate artistic light source or an overly conservative threshold; inspect the original, output, and maps. The repair is a single blend toward the input and may leave failures unresolved. A missing reference means PSNR/SSIM/LPIPS cannot be computed. A null no-reference metric means its model or required data was unavailable, not a score of zero.

## Before a final report

1. Run classical, learned, and hybrid modes on paired test images and at least one unseen unpaired set.
2. Run each hybrid ablation and report its image count, mean/median metric, variability, and failures.
3. Review lowest quality and highest-latency samples, all unresolved guardrails, and every model fallback.
4. Record environment, config digest, checkpoint hash, device, precision, resize policy, dataset version, and split.
5. Report latency on the intended presentation machine after a documented warm-up. Do not use synthetic gallery images as evaluation data.
