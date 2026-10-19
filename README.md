# Kidney Stone Ultrasound CNN (Normal vs Stone)

    pip install -r requirements.txt
    unzip archive.zip -d data
    python train.py --data data --epochs 25

Outputs in `results/`: `best.keras`, `curves.png`, `metrics.json`, plus a printed
classification report and confusion matrix.

## Dataset findings (checked on your archive)
- 9,416 images, 512x512 RGB: Normal 4,414 / Stone 5,002.
- 1,040 exact duplicate files (mostly Stone) -> removed before splitting.
- ~28% of sampled Stone images have colored calipers/text; 0% of Normal do.
  Left alone, the CNN can score high by spotting the overlay rather than the
  stone. `train.py` inpaints colored pixels and uses grayscale.
- Adjacent file numbers are often near-identical frames -> block-wise split.

## Caveats
- White overlays (scale bar, text) can't be removed by color; if accuracy is
  suspiciously >99%, crop the image borders or run Grad-CAM to check where
  the model looks.
- Frames likely come from a limited set of patients and no patient IDs are
  given, so the test score may still be optimistic.
- Research/learning project only, not for clinical use.

## Anti-overfitting options (train.py flags)
    python train.py --data data                                   # small CNN, dropout 0.5, L2, label smoothing
    python train.py --data data --model small --crop 16           # also crop border text/scale bar
    python train.py --data data --model transfer --finetune 10    # MobileNetV2 pretrained, then fine-tune
Other flags: `--dropout`, `--l2`, `--label-smoothing`, `--patience`, `--weights none` (offline).
After training, check the printed train-vs-val gap and `results/curves.png`.
