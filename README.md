# Collective Variational Auto-Encoder (cVAE)
- pytorch implementation
- sample data for running

Please kindly cite our article if you use this repository

```
@inproceedings{DBLP:conf/recsys/ChenR18,
  author    = {Yifan Chen and
               Maarten de Rijke},
  title     = {A Collective Variational Autoencoder for Top-N Recommendation with
               Side Information},
  booktitle = {Proceedings of the 3rd Workshop on Deep Learning for Recommender Systems,
               DLRS@RecSys 2018, Vancouver, BC, Canada, October 6, 2018},
  pages     = {3--9},
  year      = {2018},
  crossref  = {DBLP:conf/recsys/2018dlrs},
  url       = {https://doi.org/10.1145/3270323.3270326},
  doi       = {10.1145/3270323.3270326},
  timestamp = {Wed, 21 Nov 2018 12:44:01 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/recsys/ChenR18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

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
