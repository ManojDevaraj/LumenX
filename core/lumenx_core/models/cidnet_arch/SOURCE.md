# Official HVI-CIDNet architecture

Source: https://github.com/Fediory/HVI-CIDNet/tree/master/net
Copyright (c) 2024 Fediory Feng. MIT licence retained in LICENSE.

Fetched official Git blobs:

- CIDNet.py: 134bd925d4a2fd9927820d1529071690768b96f4
- HVI_transform.py: 05233b7e7cee610435fed7c68460cc42947b5c1b
- transformer_utils.py: fd3a4ba19ce068faabac0d600ecc261599103dd0
- LCA.py: 474480164b850a3422d4f82aedfa5ed31ba1c81b

Integration changes: package-relative imports; remove only the unused remote
PyTorchModelHubMixin dependency so inference never contacts a model hub. Model
layers and state-dict names are unchanged. Wrappers own padding, precision,
checkpoint verification and tiling. No weights are vendored in the package.
Trailing whitespace was normalized without changing source statements.

Precision correction: compute HVI forward/inverse colour math and attention vector
normalization in float32, then restore the incoming dtype. Official epsilon values
such as 1e-8/1e-12 underflow in FP16 at black/flat inputs. Convolution/attention
inference remains FP16 on CUDA; architecture/state-dict names are unchanged.
