"""Locked controls. The report states these rules before any score is read."""

SEED = 20261019
CATEGORIES = (
    "Normal",
    "Reconnaissance",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
)
WORMS = "Worms"

# Tuning split of the deduplicated table. The last portion is never scored.
TUNE_FRACTIONS = (0.70, 0.15, 0.15)
N_FOLDS = 5

MAP_SIZES = ((8, 8), (12, 12), (16, 16), (20, 20), (24, 24), (28, 28), (32, 32), (36, 36), (40, 40))
TRANSFORM_SHAPE = (16, 16)
# Smallest map whose validation macro F lies within this band of the best.
SIZE_BAND = 0.01

ORDER_EPOCHS = 8
BATCH_EPOCHS = 25
QE_TOL = 1e-4
QE_PATIENCE = 5
ETA0 = 0.5
# Convergence keeps a one-cell Gaussian width so a neighbourhood vote
# remains a neighbourhood. The notes allow the width to fall to zero.
SIGMA_CONV = 1.0
CLIP = 3.0

# One-at-a-time sensitivity of the fixed training settings on the locked map.
# An alternative replaces the default only when it beats the default by more
# than the seed spread.
DEFAULTS = {"eta0": ETA0, "order_epochs": ORDER_EPOCHS, "sigma_conv": SIGMA_CONV}
SENSITIVITY = (
    ("eta0", 0.1),
    ("eta0", 0.9),
    ("order_epochs", 4),
    ("order_epochs", 16),
    ("sigma_conv", 0.5),
    ("sigma_conv", 2.0),
)

CLUSTER_COUNTS = (4, 6, 8, 10, 12, 16, 20, 30, 50, 100, 200)
CAP = 2000
WORMS_RECALL_TRIGGER = 0.2
# Growth may add at most this many rows and columns to the largest swept map.
GROW_STEPS = 6
GROW_GAMMA = 0.5

K_NEIGHBOURS = 7
TREE_DEPTH = 15
TREE_MIN_LEAF = 1

RARE_MIN = 50
SKEW_ABS = 2.0

RAW_ROWS = 257_673
DEDUP_ROWS = 162_745
