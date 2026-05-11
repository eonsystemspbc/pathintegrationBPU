# Data Validation

- PASS: neurons.csv exists
- PASS: roi_counts.csv exists
- PASS: connections.csv exists
- PASS: all queried connectome neurons are retained in pool_assignments.csv (139255/139255)
- PASS: pool assignment is exhaustive and mutually exclusive
- Pool counts: `{'internal': 125329, 'output': 6963, 'sensory': 6963}`
- PASS: graph metadata N matches neuron export (139255 neurons)
- PASS: no train/val/test leakage across 9 cached split files
