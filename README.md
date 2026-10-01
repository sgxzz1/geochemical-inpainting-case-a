# Case A experiment review package

This package contains both results and runnable source code for three models:

1. `no_prior_generation/`: direct conditional diffusion without DS/Kriging priors.
2. `no_prior_direct_prediction/`: deterministic no-prior direct prediction.
3. `dual_gate_dual_prior_model/`: DS + Kriging dual-gate deterministic prediction.

Each model directory has a `code/` folder with its complete flat Python module
set and a README containing commands that use paths inside this package.

Original source roots are recorded in `manifest.json`.  The package was
generated from the Case A experiment and does not alter the original files.
