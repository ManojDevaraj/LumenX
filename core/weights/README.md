# Model weights

Optional research checkpoints are stored locally here and must not be committed. Set `LUMENX_WEIGHTS_DIR` to use another directory.

The supported learned model in this build is Zero-DCE++. Download the authors' `Epoch99.pth` checkpoint with:

```powershell
python -m lumenx_core.download_weights
```

The downloader prints a SHA-256 digest. The upstream implementation and checkpoint are restricted to academic, non-commercial use unless separately licensed; see [`../LICENSE_NOTICES.md`](../LICENSE_NOTICES.md). The checkpoint is ignored by Git and is not bundled with the project.
