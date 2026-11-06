# Dataset acquisition and layout

The repository intentionally does not contain third-party datasets. Download datasets from their source/organizer pages, review their terms, keep them outside version control, and pass local paths to the evaluator.

| Dataset | Use | Source |
| --- | --- | --- |
| LOL / LOL-v1 | Paired low/normal-light evaluation | [Authors' BMVC 2018 page](https://daooshee.github.io/BMVC2018website/) |
| LOL-v2 Real / Synthetic | Paired evaluation, with the published test split held out | [LOL-v2 dataset project](https://github.com/Elin24/Awesome-Low-Light-Enhancement) |
| DICM, LIME, MEF, NPE, VV | Unpaired, previously unseen low-light evaluation | [Dataset collection linked by the benchmark literature](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T) |
| MIT-Adobe FiveK | Paired exposure/style study; select a documented reference style | [MIT CSAIL FiveK](https://data.csail.mit.edu/graphics/fivek/) |

Common paired layouts are sibling directories named `Low` and `High` or `Low` and `Normal`. For other layouts, supply `--input-dir` and `--reference-dir`. The evaluator matches relative paths first, then filenames without extensions. Paired references with different dimensions are resized to the processed input geometry and this choice should be recorded in the report.

Example:

```powershell
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LOL-v2-Real --root D:\datasets\LOL-v2\Real_captured\Test --input-dir D:\datasets\LOL-v2\Real_captured\Test\Low --reference-dir D:\datasets\LOL-v2\Real_captured\Test\Normal --paired
```

Avoid overlap between training and benchmark splits. Do not fine-tune on benchmark reference images. For unseen sets without aligned normal-light references, omit `--paired`; report no-reference metrics and custom changes only. Maintain a record of dataset release, subset, source, split, and any excluded or unreadable files.

## LOL-v1 reproduction

The authors' archive used for the local run contains 485 training pairs under `our485/low` and `our485/high`, plus 15 held-out pairs under `eval15/low` and `eval15/high`. Keep this third-party archive and extracted images local; the repository ignores the contents of this directory. The evaluation below uses only the held-out split.

Run the three required approaches on all 15 pairs:

```powershell
$env:LUMENX_ZERODCE_CHECKPOINT = ""
$env:LUMENX_WEIGHTS_DIR = (Resolve-Path "core/weights").Path
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LOL-v1-eval15 --root server\datasets\lol_v1\eval15 --input-dir server\datasets\lol_v1\eval15\low --reference-dir server\datasets\lol_v1\eval15\high --paired --modes classical,dl,hybrid --output server\results
```

Optional no-reference fine-tuning uses only the training low-light images. The one-epoch command below is a short CPU adaptation run; it is a research checkpoint that must be compared with the upstream model before deployment:

```powershell
core\.venv\Scripts\python.exe core\scripts\train_zerodce.py --data server\datasets\lol_v1\our485\low --output server\datasets\zerodce_lol_v1_epoch1.pth --init core\weights\zerodce_pp.pth --epochs 1 --batch-size 8 --size 504 --device cpu
$env:LUMENX_ZERODCE_CHECKPOINT = (Resolve-Path "server/datasets/zerodce_lol_v1_epoch1.pth").Path
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LOL-v1-eval15-ft-1epoch --root server\datasets\lol_v1\eval15 --input-dir server\datasets\lol_v1\eval15\low --reference-dir server\datasets\lol_v1\eval15\high --paired --modes dl,hybrid --output server\results
```

Run component ablations and refresh the tables and chart:

```powershell
$env:LUMENX_ZERODCE_CHECKPOINT = ""
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LOL-v1-eval15-ablation --root server\datasets\lol_v1\eval15 --input-dir server\datasets\lol_v1\eval15\low --reference-dir server\datasets\lol_v1\eval15\high --paired --modes hybrid --ablations no_tuner,no_fusion,no_classical,no_dl,no_gain_denoise,no_recoverability,no_guardrails,no_color_correction --output server\results
core\.venv\Scripts\python.exe server\eval\aggregate.py --results server\results
core\.venv\Scripts\python.exe server\eval\mine_failures.py --results server\results
```

An unpaired generalization check can use images from the linked LIME/NPE collection. Omit `--paired` and the reference directory; paired PSNR/SSIM will be null while the engine's no-reference custom measurements and audits are retained:

```powershell
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LIME-unpaired-10 --root server\datasets\unpaired\LIME --input-dir server\datasets\unpaired\LIME --modes classical,dl,hybrid --output server\results
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset NPE-unpaired-8 --root server\datasets\unpaired\NPE --input-dir server\datasets\unpaired\NPE --modes classical,dl,hybrid --output server\results
core\.venv\Scripts\python.exe server\eval\aggregate.py --results server\results
```

The evaluator writes resumable JSONL rows, a progress file, aggregate CSV/JSON/SVG, and a failure-review CSV. Optional learned IQA metrics remain null unless their models are installed; the evaluator records those unavailable metrics in each row.
