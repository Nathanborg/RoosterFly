<p align="center">
  <img src="exp_realism/assets/roosterfly_logo.png" alt="RoosterFly" width="360">
</p>

# RoosterFly

**Interactive root segmentation and annotation for minirhizotron images.** A deep-learning model
predicts the roots of every frame; you check and correct them with tools that follow the roots
themselves; corrections travel between the dates of the same tube and window; and the model can be
retrained on your corrections.

## Features

**Prediction**
- U-Net (MiT-B3 encoder) prediction with the exact training-time preprocessing: per-image histogram
  matching to the training reference and a 4th input band (G − B − 2R).
- Background queue: the image on screen first, then the next ones ("Predict ahead"), or the whole
  folder ("Pre-predict Folder"). Maps are cached on disk, so a folder is predicted once.
- Adjustable threshold, test-time augmentation, camera border bars masked, "Ignore edges".

**Correcting**
- **Smart Brush**: a stroke (or a click) snaps onto the root's crest, follows it past the stroke until
  it fades, bends away or meets another root, and measures its real width. Brushing over a root
  re-measures and widens it. An optional colour cue learns the colour of your roots and soil.
- **Smart Eraser**: removes the full width of the root under the stroke and an over-followed tail
  (Shift: exact brush area; Ctrl: whole connected root). **Manual Pen**, undo / redo.
- **Delete all**: every edit of the folder (project) at once, back to the model's prediction only. The
  saved files are moved to `_annotations/_deleted_<date-time>`, not destroyed.

**Dates of the same tube & window**
- Frames named like `..._T61_L004_2018.12.11_103144_3_...` or `CD_T017_L002_02.06.19_141800_020_CA`
  (CI-600) are grouped by site, tube and window and ordered by date.
- **Copy to prev / next**, **Accept prev. date**, **Find in other dates…**: the annotation is moved onto
  another date at the exact spot. The frames are registered (phase correlation on a small rotation
  grid, then StackReg), each root is matched again around itself, and finally slid onto the root in
  the new image (ridge and model evidence). Onto an image that already has polygons you can add only
  the missing roots, or replace them.
- The previous date's roots can be shown (cyan) or re-traced on the new date automatically.

**Learning from your corrections**
- **Train on my corrections…**: fine-tunes the decoder of the model on your saved masks (encoder frozen),
  on the CPU here or with a generated Google Colab notebook on a GPU. The new model joins the Model list.
- Optional synthetic training images from the built-in root simulator, fitted to the widths, lengths
  and colours of your annotated roots, rendered on your own soil.

**Separating touching roots**
- The model has a second output, a centre-line map (1 in the middle of every root), which tells touching
  and crossing roots apart. **Separate roots** (Display View) shows every root in its own colour with its
  centre line: centre lines are cut at junctions and the pieces that continue straight through a crossing
  are joined, then every root pixel goes to its root. On a dense CI-600 scan: 2,001 roots where the mask
  had 627 blobs.
- **Length from: Centre lines** (default) measures along these centre lines - the skeleton of a clump grows
  spurious spokes (22 % more length on that scan). Hand-drawn shapes use their own shape.

**Results**
- Each saved image: `_annotations/<image>_mask.png`, an overlay, and a row in `traits_summary.csv`.
- **Export CSV…**: one row per image with its name, site / tube / window / date / time / session and
  the roots' length, projected and surface area, volume, mean diameter, coverage, length density,
  endpoints, branch points and length by diameter class (0 for images without roots) - or **one row per
  root** (separated roots: length, diameter, area, surface, volume, position) - for the annotated
  images, or for all images with the model's prediction where you have not annotated.

## Install

Python 3.8 or newer.

```bash
git clone https://github.com/Nathanborg/RoosterFly.git
cd RoosterFly
pip install -r requirements.txt
```

`torch` installs the CPU build by default; for a GPU see [pytorch.org](https://pytorch.org/get-started/locally/).

Put a trained model in `segmentation/realsoil/<run name>/` (see
[segmentation/realsoil/README.md](segmentation/realsoil/README.md)), or point `--root` at a project
folder that has one.

## Run

```bash
python roosterfly.py --dir "path/to/image/folder"
```

| Option | |
|---|---|
| `--dir`, `--image` | folder or single image to open |
| `--model` | run name in `segmentation/realsoil` (default `v8_full_real_and_syn_mitb3`) |
| `--root` | project folder holding `segmentation/realsoil` (default: this folder) |
| `--px-per-mm` | image scale for the traits (default 11.81 = 300 dpi) |
| `--lookahead`, `--predict-all` | background prediction of the next images / the whole folder |

## Keys

| Key | | Key | |
|---|---|---|---|
| A / D | previous / next image | B / E / M / H | brush / eraser / pen / pan |
| Enter | save & next | `[` `]` | smaller / larger brush |
| Ctrl+S | save | Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+Shift+A / D | copy to previous / next image | P | accept the previous date |
| Shift + stroke | stroke only (no following) | Space | fit view |
| Wheel | zoom | Wheel pressed / right button + drag | pan |

## Files written next to your images

```
<image folder>/_annotations/
    <image>_mask.png  <image>_overlay.jpg     saved annotations
    traits_summary.csv                        one row per saved image
    roots_export_<date>[_roots].csv           Export CSV (per image / per root)
    _prob_cache/<model>/, <model>_cl/         cached predictions and centre-line maps
    _deleted_<date-time>/                     what "Delete all" removed
```

## License

MIT - see [LICENSE](LICENSE). © 2026 Nathan B. Gonçalves.
