# Test Samples

This folder contains a minimal set of sample hand landmark data
used by `test_system.py`.

If you wish to run the model prediction test with different data,
replace `sample_landmarks.json` with your own landmarks in the
same format:

```
[
  {"label": "A", "landmarks": [63 floating point values]},
  {"label": "B", "landmarks": [63 floating point values]}
]
```

The provided samples use synthetic values so they do not correspond to
actual hand positions. They are only intended for sanity checking that the
prediction pipeline works.
