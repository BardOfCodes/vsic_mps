# VSIC 2024: Marching Primitive Starter Kit

The repository adds a small wrapper on the [marching-primitives](https://arxiv.org/abs/2303.13190) [codebase](https://github.com/ChirikjianLab/Marching-Primitives) released by the authors for the [Visual Shape Inference Challenge](https://github.com/BardOfCodes/vsic).

## Steps

1. Download the dataset from [3DCoMPaT](https://github.com/Vision-CAIR/3DCoMPaT) repository. Unzip the 3DCoMPaT dataset to get the `.gtlf` mesh files.
2. Download the `vsic_split.json` from the [vsic](https://github.com/BardOfCodes/vsic) repository.
3. Run the `run_mps.py` script with the relevant args.

This will run the system on all the shapes in the `test` split. The output will be saved as `final_expressions.pkl`. This file can be submitted to the evaluation server directly.
