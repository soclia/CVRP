# HACD

PyTorch implementation of HACD for the Capacitated Vehicle Routing Problem (CVRP).
The model combines HiLo encoding, structural anchors and capacity-aware residual decoding.

## Environment

- Python 3.10 or later
- PyTorch 2.4 or later (below 3.0)
- NumPy, SciPy and VRPLIB

```sh
pip install -r requirements.txt
```

## Running

Generate validation and test data:

```sh
python generate_data.py --split validation --episodes 1000 --seed 271828
python generate_data.py --split test --episodes 10000 --seed 314159
```

Training:

```sh
python train.py --validation_data data/cvrp100_validation_seed271828.pt
```

Testing:

```sh
python test.py --checkpoint runs/full-cvrp100-seed1234/best.pt --test_data data/cvrp100_test_seed314159.pt
```

## Parameters and data

Default parameters are defined in `src/__init__.py`. Use `python train.py --help`
or `python test.py --help` to view command-line options.

The 19 CVRPLIB instances are in `data/cvrplib_instances/` and can be evaluated
with `run_cvrplib.py`.

## Acknowledgements

The implementation builds on [POMO](https://github.com/yd-kwon/POMO).
