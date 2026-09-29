# Data preparation

Install `requirements-data.txt` for generation. The verified package version is
**DeepMIMO 4.0.0b11**. The local scenario metadata records conversion version
**4.0.0a3**; the conversion and Python-package versions are distinct.

| Group | Scenario names |
| --- | --- |
| train | `city_0_newyork_28`, `city_1_losangeles_28`, `city_2_chicago_28` |
| test | `city_6_miami_28` |
| generalization | `city_6_miami_28`, `city_10_austin_28`, `city_13_columbus_28`, `city_8_dallas_28` |

All scenes use 28 GHz. The 3.5 GHz Austin scene is not required. List a group
without downloading with `python -m scripts.download_scenarios --group all --list`.
Existing scene directories are reused. The downloader uses DeepMIMO's official
`download` API; scenes remain external to this repository.

Official resources:
- [DeepMIMO database API](https://www.deepmimo.net/docs/api/database.html)
- [DeepMIMO project](https://github.com/DeepMIMO/DeepMIMO)

## Selection and generation

The generator preserves the original `dm.load(scenario)[1]` dataset-key selection.
It retains the selected dataset's receiver ordering, first applies
`get_active_idxs()`, and then keeps users whose first path's power is strictly
greater than -140 dBm. It does not replace that test with total received power.
The returned receiver indices are saved in each generated channel file's JSON
sidecar. Dataset-key semantics should be checked when migrating DeepMIMO versions.

The BS array is 20 by 5, the UE array is 2 by 2, both with half-wavelength
spacing and zero rotation. Frequency-domain generation uses 40 subcarriers,
20 MHz bandwidth, and selected subcarrier index 0. Doppler is disabled.
Other system parameters are recorded in `configs/data.json`.

Training channel arrays are concatenated in New York, Los Angeles, Chicago
order. The reference archive has 28,595 training channel realizations. For each
evaluation instance i, the generator uses seed 42+i, samples K uniformly from
3,4,5, and selects K distinct channel indices using `torch.randperm`.
The main Miami test set has 1,000 instances; each generalization city has 100.
These are distinct datasets even when the same Miami scene is used.

Scene metadata hashes, byte counts, and conversion versions in
`configs/scenarios.json` describe the author's local files. They do not pin a
future server download by themselves. If a fresh download differs, retain its
metadata and compare generated channels before claiming the same input data.

## File formats

Channel `.pt` files in the prepared archive contain tensors, with shape
`(users, UE antennas, BS antennas, selected subcarriers)` and the original
numerical precision. This replaces NumPy-array pickle payloads without changing
their values. `parameters.pt` contains system parameters; `probInsts_*.pt`
contain lists of channel matrices and user counts. Newly generated instance
files also save selected channel indices.

The default loader uses `torch.load(..., map_location="cpu", weights_only=True)`.
Use the converted release files or regenerate channels with the supplied tool;
old NumPy-pickle channel files are not accepted by this loader.

Existing output files are reused by default. Use a separate data directory when
changing configurations; the generator refuses inconsistent system parameters.
`--overwrite` explicitly regenerates matching-config outputs. A cached channel
file should only be reused with the configuration that produced it.

The optional generated-data archive is prepared separately from source. Its
redistribution terms need to be checked against the applicable DeepMIMO scenario
terms before publication. The raw scenario archive is not redistributed here.
