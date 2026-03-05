# Pretrained Models

This directory contains pretrained model checkpoints used by MOSS.

## Current Models

### lsd_mtlsd_checkpoint.pth (15 MB)
- **Architecture:** LSD Boundary 2D (MtLSD)
- **Purpose:** Membrane/boundary prediction for EM images
- **Training:** Pretrained on EM data
- **Used by:** `lsd_boundary_2d` architecture
- **No additional training needed** - ready to use

## Usage

The LSD pretrained model is automatically loaded when you:
1. Go to "Ground Truth" page
2. Select "lsd_boundary_2d" architecture
3. Model loads pretrained weights automatically

## Adding New Pretrained Models

To add a new pretrained model:
1. Place the `.pth` checkpoint file in this directory
2. In your architecture file (e.g., `models/architectures/your_model.py`), set:
   ```python
   PRETRAINED_CHECKPOINT = str(Path(__file__).parent.parent.parent.parent / 'pretrained_models' / 'your_model.pth')
   ```
3. The architecture loader will automatically detect it
