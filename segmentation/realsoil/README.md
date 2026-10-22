# Models

RoosterFly looks for trained segmentation models here, one folder per run
(the folders are not stored in git: a checkpoint is ~190 MB).

```
segmentation/realsoil/<run name>/
    seg_best.pt                 weights (bare state_dict)
    train_config.json           architecture, encoder, input mode
    prep_stats.json             histogram-matching reference (CDF_REF) and index-band statistics
    inference_settings.json     default threshold, border masking
```

The default run is `v8_full_real_and_syn_mitb3` (U-Net, MiT-B3 encoder, 4 input channels:
hist-matched RGB plus the G - B - 2R index band). Pick another one with `--model <run name>`, or keep
the models in another project with `--root <that project folder>`.

Models fine-tuned in the app ("Train on my corrections…") are written here as `v8ft_<date>_<n>img`
and appear in the Model list after pressing ↻.
