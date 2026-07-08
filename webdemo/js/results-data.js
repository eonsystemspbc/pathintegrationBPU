/* AUTO-GENERATED from real result files by scratchpad/build_data.py. Do not edit by hand. */
window.RESULTS = {
 "cxHeading": {
  "T": [
   50,
   100,
   200
  ],
  "frozen": {
   "cx_bpu": {
    "mean": [
     1.0536,
     1.2995,
     1.4407
    ],
    "std": [
     0.0032,
     0.0016,
     0.0007
    ]
   },
   "weight_shuffle": {
    "mean": [
     1.0598,
     1.3088,
     1.4437
    ],
    "std": [
     0.0071,
     0.0058,
     0.002
    ]
   },
   "degree_shuffle": {
    "mean": [
     1.1998,
     1.3733,
     1.4777
    ],
    "std": [
     0.0024,
     0.0012,
     0.0007
    ]
   },
   "random": {
    "mean": [
     1.1598,
     1.3534,
     1.4675
    ],
    "std": [
     0.0086,
     0.0051,
     0.0023
    ]
   },
   "no_recurrence": {
    "mean": [
     1.1673,
     1.3574,
     1.4694
    ],
    "std": [
     0.0017,
     0.0008,
     0.0005
    ]
   }
  },
  "trainable": {
   "cx_bpu": {
    "mean": [
     0.4354,
     0.8012,
     1.1339
    ],
    "std": [
     0.0143,
     0.0134,
     0.0156
    ]
   },
   "weight_shuffle": {
    "mean": [
     0.4513,
     0.8496,
     1.1856
    ],
    "std": [
     0.0172,
     0.049,
     0.053
    ]
   },
   "degree_shuffle": {
    "mean": [
     0.499,
     0.904,
     1.2123
    ],
    "std": [
     0.0373,
     0.0784,
     0.0579
    ]
   },
   "random": {
    "mean": [
     0.5243,
     0.977,
     1.2675
    ],
    "std": [
     0.051,
     0.0762,
     0.0465
    ]
   },
   "no_recurrence": {
    "mean": [
     1.1538,
     1.3523,
     1.4657
    ],
    "std": [
     0.0002,
     0.0001,
     0.0
    ]
   }
  }
 },
 "opticFlow": {
  "fraction": [
   5,
   10,
   15,
   20,
   30,
   50,
   75,
   100
  ],
  "families": {
   "sparse_connectome": {
    "mean": [
     0.2196,
     0.1851,
     0.1761,
     0.1697,
     0.1591,
     0.1471,
     0.1365,
     0.1317
    ],
    "std": [
     0.0139,
     0.0044,
     0.0022,
     0.0013,
     0.0013,
     0.0003,
     0.0011,
     0.0006
    ]
   },
   "sparse_random": {
    "mean": [
     0.2057,
     0.1901,
     0.1798,
     0.1732,
     0.1659,
     0.1629,
     0.1508,
     0.1425
    ],
    "std": [
     0.0082,
     0.0038,
     0.0005,
     0.0007,
     0.0004,
     0.0023,
     0.0011,
     0.0014
    ]
   },
   "pruned_connectome": {
    "mean": [
     0.2036,
     0.1807,
     0.1755,
     0.169,
     0.1594,
     0.1488,
     0.1392,
     0.1317
    ],
    "std": [
     0.0061,
     0.0007,
     0.0017,
     0.0008,
     0.0002,
     0.0007,
     0.0014,
     0.0009
    ]
   },
   "pruned_random": {
    "mean": [
     0.2044,
     0.1913,
     0.1782,
     0.1724,
     0.1646,
     0.1554,
     0.1474,
     0.1392
    ],
    "std": [
     0.0043,
     0.0085,
     0.0018,
     0.0009,
     0.0011,
     0.0017,
     0.0021,
     0.0004
    ]
   },
   "dense_connectome": {
    "mean": [
     0.1909,
     0.1697,
     0.162,
     0.1546,
     0.1449,
     0.1356,
     0.1293,
     0.1251
    ],
    "std": [
     0.0049,
     0.0006,
     0.0007,
     0.0002,
     0.0007,
     0.0003,
     0.0004,
     0.0006
    ]
   },
   "dense_random": {
    "mean": [
     0.1861,
     0.1705,
     0.1622,
     0.1549,
     0.1445,
     0.1359,
     0.1296,
     0.1254
    ],
    "std": [
     0.001,
     0.0017,
     0.0004,
     0.0001,
     0.0003,
     0.0006,
     0.0006,
     0.0002
    ]
   }
  }
 },
 "bpu": {
  "mnist": [
   {
    "model": "Connectome",
    "acc": 0.9653,
    "std": 0.0009,
    "kind": "bio"
   },
   {
    "model": "Random sparse",
    "acc": 0.9668,
    "std": 0.0006,
    "kind": "ctrl"
   },
   {
    "model": "Weight shuffle",
    "acc": 0.9665,
    "std": 0.0021,
    "kind": "shuffle"
   },
   {
    "model": "Dense trainable",
    "acc": 0.9705,
    "std": 0.001,
    "kind": "dense"
   },
   {
    "model": "MLP (no recurrence)",
    "acc": 0.9708,
    "std": 0.0003,
    "kind": "mlp"
   }
  ],
  "cifar": [
   {
    "model": "Connectome",
    "acc": 0.4682,
    "std": 0.0065,
    "kind": "bio"
   },
   {
    "model": "Random sparse",
    "acc": 0.491,
    "std": 0.0047,
    "kind": "ctrl"
   },
   {
    "model": "Weight shuffle",
    "acc": 0.485,
    "std": 0.0013,
    "kind": "shuffle"
   },
   {
    "model": "Dense trainable",
    "acc": 0.5475,
    "std": 0.0053,
    "kind": "dense"
   },
   {
    "model": "MLP (no recurrence)",
    "acc": 0.4927,
    "std": 0.0093,
    "kind": "mlp"
   }
  ]
 },
 "continual": {
  "models": [
   {
    "name": "Connectome",
    "acc": 0.8189,
    "acc_se": 0.0012,
    "forget": 0.2234,
    "kind": "bio"
   },
   {
    "name": "Weight shuffle",
    "acc": 0.7873,
    "acc_se": 0.004,
    "forget": 0.2617,
    "kind": "shuffle"
   },
   {
    "name": "Random",
    "acc": 0.7476,
    "acc_se": 0.0066,
    "forget": 0.308,
    "kind": "ctrl"
   }
  ],
  "retentionCurve": {
   "stage": [
    1,
    2,
    3,
    4,
    5
   ],
   "connectome": [
    0.999,
    0.9412,
    0.8784,
    0.8422,
    0.8189
   ],
   "weight_shuffle": [
    0.999,
    0.9317,
    0.8644,
    0.8181,
    0.7873
   ],
   "random": [
    0.993,
    0.9133,
    0.8499,
    0.8069,
    0.7476
   ]
  },
  "Rmatrix": {
   "connectome": [
    [
     0.999,
     0.886,
     0.768,
     0.767,
     0.714
    ],
    [
     0.475,
     0.996,
     0.87,
     0.747,
     0.796
    ],
    [
     0.428,
     0.488,
     0.997,
     0.856,
     0.798
    ],
    [
     0.512,
     0.476,
     0.471,
     0.999,
     0.789
    ],
    [
     0.43,
     0.457,
     0.464,
     0.41,
     0.998
    ]
   ],
   "weight_shuffle": [
    [
     0.999,
     0.868,
     0.749,
     0.739,
     0.671
    ],
    [
     0.495,
     0.995,
     0.85,
     0.704,
     0.748
    ],
    [
     0.441,
     0.492,
     0.994,
     0.831,
     0.757
    ],
    [
     0.504,
     0.465,
     0.461,
     0.998,
     0.763
    ],
    [
     0.429,
     0.452,
     0.457,
     0.411,
     0.997
    ]
   ],
   "random": [
    [
     0.993,
     0.834,
     0.694,
     0.68,
     0.561
    ],
    [
     0.469,
     0.992,
     0.864,
     0.719,
     0.705
    ],
    [
     0.429,
     0.496,
     0.992,
     0.832,
     0.727
    ],
    [
     0.497,
     0.478,
     0.462,
     0.996,
     0.748
    ],
    [
     0.419,
     0.448,
     0.44,
     0.374,
     0.997
    ]
   ]
  }
 },
 "reversal": [
  {
   "name": "Connectome",
   "recall": 0.9925,
   "epoch95": 9.5,
   "kind": "bio"
  },
  {
   "name": "Weight shuffle",
   "recall": 0.9912,
   "epoch95": 10.5,
   "kind": "shuffle"
  },
  {
   "name": "Random sparse",
   "recall": 0.9735,
   "epoch95": 22.0,
   "kind": "ctrl"
  },
  {
   "name": "Degree-preserving",
   "recall": 0.9632,
   "epoch95": 33.5,
   "kind": "ctrl"
  }
 ],
 "mqar": [
  {
   "model": "Attention + short-conv",
   "acc": 1.0,
   "kind": "ceiling"
  },
  {
   "model": "Connectome \u00b7 1000 ep",
   "acc": 0.995,
   "kind": "bio"
  },
  {
   "model": "Connectome \u00b7 200 ep",
   "acc": 0.925,
   "sd": 0.003,
   "kind": "bio"
  },
  {
   "model": "Weight shuffle",
   "acc": 0.914,
   "sd": 0.003,
   "kind": "shuffle"
  },
  {
   "model": "Random sparse",
   "acc": 0.836,
   "sd": 0.008,
   "kind": "ctrl"
  },
  {
   "model": "Degree-preserving",
   "acc": 0.768,
   "sd": 0.033,
   "kind": "ctrl"
  },
  {
   "model": "Chance",
   "acc": 0.031,
   "kind": "chance"
  }
 ]
};
