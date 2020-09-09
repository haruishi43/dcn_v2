# DCNv2

This is a fork from [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2) that just improves on the build system so users can just `pip install .` and have DCNv2 globally.
I also did some check so that C++/CUDA codes are up-to-date with the pytorch api.

## Installation:

- Install pytorch >= 1.5.0 from [here](https://pytorch.org/get-started/previous-versions/#linux-and-windows-3)

```Bash
git clone https://github.com/haruishi43/dcn_v2
cd dcn_v2
pip install .
```

You can uninstall `DCNv2` using `pip uninstall DCNv2`

## Usage:

See some of the `tests` scripts for ideas.

```Python
from dcn_v2 import DCN

input = torch.randn(2, 64, 128, 128)
# wrap all things (offset and mask) in DCN
dcn = DCN(
    64, 64, kernel_size=(3, 3), stride=1,
    padding=1, deformable_groups=2)
```
