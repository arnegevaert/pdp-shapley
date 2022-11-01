# Reproducing experiments

Install the development requirements:
```bash 
 pip install -e .[dev]
```

## Reproducing synthetic data experiments

## Reproducing real data experiments

**Step 1.** Use `prerequisites.py` to compute experiment prerequisites for a given dataset.
  for more info, run `prerequisites.py -h`. 

**Step 2.** Run an experiment using `experiment.py`. For more info, run `experiment.py -h`.

**Step 3.** Evaluate the results using `evaluate.py`. For more info, run `evaluate.py -h`.
