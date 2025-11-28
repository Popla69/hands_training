# FreiHAND Dataset (Converted)

## Statistics
- Total samples: 32560
- Training samples: 29304
- Validation samples: 3256

## Format
- Images: data/freihand_converted/images/
- Annotations: JSON files with 21 landmarks per hand
- Landmark format: [[x1, y1, z1], [x2, y2, z2], ...]

## Usage
```bash
python hand_landmark_v2/train.py --data_dir data/freihand_converted --epochs 200
```

## Original Dataset
FreiHAND: https://lmb.informatik.uni-freiburg.de/projects/freihand/
