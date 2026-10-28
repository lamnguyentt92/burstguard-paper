"""BurstGuard: robustness of federated fine-tuning of tiny time-series foundation models."""

__version__ = "0.3.16"
VERSION_TAG = f"v{__version__}"
DATA_FORMAT = 1  # bump when the processed .npz layout changes (file names carry this number)
# Version of the experiment code (models, attacks, defenses, metrics, grids). Releases that only change tooling keep
# it, so their results can be merged without a version-mix warning (0.3.0-0.3.3 share 0.3.0; 0.3.4 changed the
# defaults: 20 rounds, strong attacker, cheaper BurstGuard search; 0.3.5: attacker back to 0.2 / 5 epochs,
# boosted neurotoxin / adaptive, one screening thread; 0.3.6 only adds SMS / call data and tooling, so it keeps 0.3.5
# and its results merge with those of 0.3.5).
EXPERIMENT_CODE = "0.3.5"
