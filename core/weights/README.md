# Official weights (not committed)

Default location is this directory, resolved from the package rather than cwd.
Override with LUMENX_WEIGHTS_DIR. The registry verifies every file before loading;
it never downloads. Run from the repository root in a network-enabled terminal:

```powershell
.\core\.venv\Scripts\python.exe core/scripts/download_weights.py
```

The source-of-truth URLs and full hashes are in
`lumenx_core/config/checkpoints.yaml`. Expected files:

| File | Publisher | SHA256 |
|---|---|---|
| cidnet_lolv2_real.pth | Fediory/HVI-CIDNet, LOLv2-real best PSNR | 44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6 |
| cidnet_sid.pth | Fediory/HVI-CIDNet, SID | 49678f69495610a2691bb5b2665d2766b57a6a79201a236f7afa84f67c5fec5f |
| zerodce_pp.pth | Li-Chongyi/Zero-DCE_extension, Epoch99 | ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84 |

HVI publisher upload with LFS hashes:
https://huggingface.co/Fediory/HVI-CIDNet/commit/98e69f687810285d3cf863745a6f78977f262ca4
HVI architecture licence: MIT; retained under models/cidnet_arch/LICENSE.
Zero-DCE++ terms: academic research use, CC BY-NC 4.0. See
https://github.com/Li-Chongyi/Zero-DCE_extension#license and the vendored NOTICE.md.

Incorrect hashes, incompatible state dictionaries, and absent model dependencies
are reported rather than loaded as if valid. No random-weight output is a valid
restoration model. Direct neural evaluation and GPU release gates require actual
publisher weights; substitutes do not satisfy those gates.
