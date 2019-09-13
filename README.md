# Collective Variational Auto-Encoder (cVAE)
- pytorch implementation
- sample data for running

## Requirement
1. python >= 3.6
1. pytorch >= 1.0
1. pytrec_eval: https://github.com/cvangysel/pytrec_eval

## Running
check specifications by 
```python
python cvae/cvae.py -h
```

### Sample run (running on GPU)
pre-train with side information
```python
python cvae/cvae.py --dir data --data music -a 10 -b 0.1 -m 50 -N 20 --layer 100 20 --save --gpu
```
refine by rating
```python
python cvae/cvae.py --dir data --data music -a 1 -b 1 -m 30 -N 20 --layer 100 20 --load 1 --rating --gpu
```

If you want to test on CPU, simply remove --gpu.
