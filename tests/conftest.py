import os

# LightGBM and PyTorch each ship their own OpenMP runtime; on macOS using both
# multi-threaded in one process can deadlock. Single-thread the tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")
