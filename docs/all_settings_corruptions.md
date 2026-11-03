# All training settings: corruption category results

[Training objectives, seeds, budgets and comparison limitations](all_training_results.md).
Each row is a completed run, identified by the same run key as the manifest.

Class-IL predicts among ten classes; Task-IL uses the correct pair. Each corruption
averages five severities; each category averages its corruptions. Overall weights
the four groups by (3,4,4,4), not equally. Groups follow MetaMix Tables 6-7:
Noise = Gaussian/shot/impulse; Blur = defocus/glass/motion/zoom;
Weather = snow/frost/fog/brightness; Digital = contrast/elastic/pixelate/JPEG.
The five two-class task structure matches the paper; exact author class order is
unverified. Our budget, architecture, seed counts and implementation differ.

All values are percentages. These are aggregates of saved full evaluations, not
new training, new corruptions, or a claim to reproduce the paper's accuracy.

## Class-IL category means

| Run | Seed | Noise | Blur | Weather | Digital | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 25.04 | 23.82 | 25.94 | 25.28 | 25.02 |
| feature_mse_seed17 | 17 | 35.76 | 43.82 | 39.34 | 39.55 | 39.88 |
| feature_mse_seed17_lambda0 | 17 | 29.80 | 38.65 | 33.76 | 34.96 | 34.59 |
| feature_mse_seed18 | 18 | 39.62 | 44.49 | 40.79 | 42.52 | 42.00 |
| feature_mse_seed18_lambda0 | 18 | 38.56 | 40.57 | 41.58 | 43.03 | 41.09 |
| feature_mse_seed19 | 19 | 38.39 | 43.39 | 41.54 | 40.37 | 41.09 |
| feature_mse_seed19_lambda0 | 19 | 33.19 | 39.33 | 35.93 | 38.47 | 36.97 |
| infonce_seed17 | 17 | 41.54 | 46.50 | 40.52 | 42.82 | 42.93 |
| classaware_infonce_seed17 | 17 | 44.81 | 48.00 | 43.11 | 45.03 | 45.27 |
| mse05_nce05_seed17 | 17 | 39.36 | 43.94 | 39.80 | 40.40 | 40.97 |
| mse02_nce1_seed17 | 17 | 38.89 | 45.72 | 40.67 | 41.98 | 42.01 |
| augce_seed17 | 17 | 40.25 | 44.94 | 42.10 | 42.24 | 42.52 |
| barlow_seed17 | 17 | 40.03 | 42.96 | 40.39 | 40.76 | 41.10 |
| bt_threepair_seed17 | 17 | 36.67 | 44.18 | 38.95 | 41.58 | 40.59 |
| multihead_seed17 | 17 | 35.15 | 38.23 | 36.08 | 35.89 | 36.42 |
| infonce_recon_seed17 | 17 | 39.53 | 44.20 | 40.08 | 41.29 | 41.39 |
| learned_js | 17 | 32.75 | 39.33 | 38.63 | 38.40 | 37.58 |
| random_js | 17 | 40.38 | 44.00 | 41.69 | 41.95 | 42.11 |
| random_infonce | 17 | 48.78 | 49.96 | 47.48 | 46.98 | 48.27 |
| learned_infonce | 17 | 36.42 | 40.35 | 40.54 | 40.59 | 39.68 |
| learned_infonce_js | 17 | 34.71 | 40.06 | 39.16 | 39.31 | 38.55 |
| chain_aware_fixed_m | 17 | 45.06 | 47.26 | 47.51 | 48.31 | 47.17 |

### Class-IL — Noise per corruption

| Run | Seed | Gauss | Shot | Impulse |
| --- | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 25.84 | 26.77 | 22.49 |
| feature_mse_seed17 | 17 | 36.22 | 38.31 | 32.73 |
| feature_mse_seed17_lambda0 | 17 | 30.77 | 32.56 | 26.06 |
| feature_mse_seed18 | 18 | 40.49 | 42.28 | 36.09 |
| feature_mse_seed18_lambda0 | 18 | 39.49 | 42.00 | 34.20 |
| feature_mse_seed19 | 19 | 39.29 | 42.08 | 33.80 |
| feature_mse_seed19_lambda0 | 19 | 33.61 | 35.89 | 30.06 |
| infonce_seed17 | 17 | 41.43 | 43.37 | 39.81 |
| classaware_infonce_seed17 | 17 | 44.61 | 46.39 | 43.43 |
| mse05_nce05_seed17 | 17 | 39.16 | 41.05 | 37.88 |
| mse02_nce1_seed17 | 17 | 38.64 | 41.10 | 36.92 |
| augce_seed17 | 17 | 40.70 | 42.58 | 37.48 |
| barlow_seed17 | 17 | 40.18 | 42.07 | 37.83 |
| bt_threepair_seed17 | 17 | 37.61 | 39.46 | 32.94 |
| multihead_seed17 | 17 | 35.97 | 37.03 | 32.45 |
| infonce_recon_seed17 | 17 | 39.38 | 41.45 | 37.76 |
| learned_js | 17 | 34.12 | 35.89 | 28.25 |
| random_js | 17 | 40.77 | 42.54 | 37.83 |
| random_infonce | 17 | 48.50 | 50.78 | 47.06 |
| learned_infonce | 17 | 37.25 | 39.50 | 32.51 |
| learned_infonce_js | 17 | 35.03 | 37.73 | 31.38 |
| chain_aware_fixed_m | 17 | 45.16 | 47.92 | 42.08 |

### Class-IL — Blur per corruption

| Run | Seed | Defocus | Glass | Motion | Zoom |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 25.64 | 23.38 | 21.72 | 24.56 |
| feature_mse_seed17 | 17 | 47.15 | 36.54 | 43.77 | 47.83 |
| feature_mse_seed17_lambda0 | 17 | 40.76 | 30.84 | 40.43 | 42.56 |
| feature_mse_seed18 | 18 | 48.22 | 37.76 | 43.67 | 48.31 |
| feature_mse_seed18_lambda0 | 18 | 43.94 | 37.89 | 39.47 | 40.97 |
| feature_mse_seed19 | 19 | 47.20 | 37.00 | 41.93 | 47.41 |
| feature_mse_seed19_lambda0 | 19 | 43.69 | 29.30 | 39.64 | 44.69 |
| infonce_seed17 | 17 | 49.02 | 42.19 | 44.94 | 49.84 |
| classaware_infonce_seed17 | 17 | 52.11 | 40.47 | 46.68 | 52.75 |
| mse05_nce05_seed17 | 17 | 47.34 | 36.57 | 43.38 | 48.45 |
| mse02_nce1_seed17 | 17 | 48.84 | 38.23 | 45.62 | 50.19 |
| augce_seed17 | 17 | 48.44 | 39.20 | 43.77 | 48.36 |
| barlow_seed17 | 17 | 47.24 | 35.79 | 41.83 | 46.97 |
| bt_threepair_seed17 | 17 | 48.54 | 35.33 | 43.72 | 49.14 |
| multihead_seed17 | 17 | 40.50 | 34.02 | 37.83 | 40.57 |
| infonce_recon_seed17 | 17 | 47.28 | 38.49 | 43.06 | 47.96 |
| learned_js | 17 | 43.32 | 35.15 | 36.27 | 42.58 |
| random_js | 17 | 47.97 | 38.16 | 42.82 | 47.05 |
| random_infonce | 17 | 53.12 | 44.62 | 49.18 | 52.92 |
| learned_infonce | 17 | 44.93 | 35.86 | 37.25 | 43.34 |
| learned_infonce_js | 17 | 44.12 | 34.97 | 37.67 | 43.48 |
| chain_aware_fixed_m | 17 | 52.44 | 40.98 | 45.35 | 50.27 |

### Class-IL — Weather per corruption

| Run | Seed | Snow | Frost | Fog | Bright |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 26.61 | 24.65 | 23.66 | 28.83 |
| feature_mse_seed17 | 17 | 36.06 | 36.41 | 41.20 | 43.71 |
| feature_mse_seed17_lambda0 | 17 | 30.19 | 30.84 | 37.30 | 36.69 |
| feature_mse_seed18 | 18 | 39.41 | 36.27 | 41.76 | 45.70 |
| feature_mse_seed18_lambda0 | 18 | 38.53 | 37.73 | 44.05 | 46.00 |
| feature_mse_seed19 | 19 | 40.77 | 38.11 | 41.66 | 45.62 |
| feature_mse_seed19_lambda0 | 19 | 33.38 | 31.33 | 38.74 | 40.28 |
| infonce_seed17 | 17 | 40.66 | 35.28 | 42.08 | 44.05 |
| classaware_infonce_seed17 | 17 | 43.32 | 37.25 | 44.04 | 47.82 |
| mse05_nce05_seed17 | 17 | 39.15 | 35.29 | 41.72 | 43.03 |
| mse02_nce1_seed17 | 17 | 39.33 | 35.40 | 43.91 | 44.05 |
| augce_seed17 | 17 | 40.69 | 37.40 | 44.41 | 45.88 |
| barlow_seed17 | 17 | 39.64 | 36.28 | 40.00 | 45.63 |
| bt_threepair_seed17 | 17 | 36.81 | 35.37 | 41.02 | 42.58 |
| multihead_seed17 | 17 | 34.16 | 33.16 | 37.26 | 39.75 |
| infonce_recon_seed17 | 17 | 40.12 | 35.10 | 41.50 | 43.61 |
| learned_js | 17 | 37.64 | 34.71 | 38.53 | 43.62 |
| random_js | 17 | 41.02 | 37.45 | 42.09 | 46.21 |
| random_infonce | 17 | 46.84 | 43.52 | 48.42 | 51.13 |
| learned_infonce | 17 | 40.22 | 36.49 | 39.48 | 45.98 |
| learned_infonce_js | 17 | 38.25 | 34.89 | 40.20 | 43.28 |
| chain_aware_fixed_m | 17 | 47.50 | 42.61 | 48.49 | 51.44 |

### Class-IL — Digital per corruption

| Run | Seed | Contrast | Elastic | Pixel | JPEG |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 22.58 | 24.63 | 26.31 | 27.59 |
| feature_mse_seed17 | 17 | 33.62 | 44.83 | 36.61 | 43.14 |
| feature_mse_seed17_lambda0 | 17 | 29.84 | 41.04 | 30.55 | 38.42 |
| feature_mse_seed18 | 18 | 37.57 | 45.94 | 40.27 | 46.29 |
| feature_mse_seed18_lambda0 | 18 | 35.69 | 46.70 | 43.36 | 46.37 |
| feature_mse_seed19 | 19 | 33.74 | 45.31 | 37.53 | 44.89 |
| feature_mse_seed19_lambda0 | 19 | 33.23 | 43.75 | 35.46 | 41.46 |
| infonce_seed17 | 17 | 35.18 | 47.37 | 43.15 | 45.60 |
| classaware_infonce_seed17 | 17 | 36.54 | 49.62 | 46.13 | 47.84 |
| mse05_nce05_seed17 | 17 | 33.61 | 45.36 | 38.93 | 43.69 |
| mse02_nce1_seed17 | 17 | 37.34 | 46.45 | 41.08 | 43.02 |
| augce_seed17 | 17 | 37.34 | 45.79 | 39.91 | 45.90 |
| barlow_seed17 | 17 | 33.97 | 44.57 | 40.02 | 44.50 |
| bt_threepair_seed17 | 17 | 33.80 | 46.71 | 41.29 | 44.51 |
| multihead_seed17 | 17 | 32.50 | 38.61 | 33.57 | 38.88 |
| infonce_recon_seed17 | 17 | 35.86 | 45.10 | 40.43 | 43.77 |
| learned_js | 17 | 31.78 | 42.22 | 36.46 | 43.13 |
| random_js | 17 | 36.30 | 45.94 | 39.71 | 45.83 |
| random_infonce | 17 | 39.44 | 50.80 | 46.85 | 50.85 |
| learned_infonce | 17 | 31.83 | 44.81 | 40.23 | 45.50 |
| learned_infonce_js | 17 | 31.98 | 43.82 | 37.23 | 44.22 |
| chain_aware_fixed_m | 17 | 38.75 | 50.65 | 51.21 | 52.62 |

## Task-IL category means

| Run | Seed | Noise | Blur | Weather | Digital | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 80.87 | 84.90 | 85.80 | 84.45 | 84.21 |
| feature_mse_seed17 | 17 | 81.08 | 84.05 | 84.90 | 84.35 | 83.76 |
| feature_mse_seed17_lambda0 | 17 | 79.28 | 81.40 | 84.73 | 83.18 | 82.34 |
| feature_mse_seed18 | 18 | 84.89 | 85.76 | 86.49 | 85.63 | 85.75 |
| feature_mse_seed18_lambda0 | 18 | 81.69 | 82.32 | 84.49 | 83.25 | 83.02 |
| feature_mse_seed19 | 19 | 82.34 | 83.99 | 85.80 | 84.09 | 84.17 |
| feature_mse_seed19_lambda0 | 19 | 79.69 | 81.36 | 85.18 | 83.88 | 82.72 |
| infonce_seed17 | 17 | 84.54 | 86.35 | 85.63 | 84.50 | 85.31 |
| classaware_infonce_seed17 | 17 | 86.01 | 87.28 | 87.28 | 86.63 | 86.85 |
| mse05_nce05_seed17 | 17 | 83.69 | 86.43 | 86.29 | 85.41 | 85.57 |
| mse02_nce1_seed17 | 17 | 83.40 | 87.95 | 87.02 | 86.28 | 86.35 |
| augce_seed17 | 17 | 83.35 | 86.27 | 86.21 | 85.67 | 85.51 |
| barlow_seed17 | 17 | 83.36 | 85.37 | 85.35 | 84.53 | 84.74 |
| bt_threepair_seed17 | 17 | 81.26 | 84.30 | 84.95 | 84.56 | 83.94 |
| multihead_seed17 | 17 | 82.42 | 84.07 | 84.32 | 83.46 | 83.65 |
| infonce_recon_seed17 | 17 | 85.27 | 87.24 | 87.01 | 86.38 | 86.56 |
| learned_js | 17 | 81.11 | 82.76 | 85.06 | 83.85 | 83.33 |
| random_js | 17 | 83.22 | 85.60 | 86.44 | 85.84 | 85.41 |
| random_infonce | 17 | 86.23 | 87.03 | 88.23 | 87.03 | 87.19 |
| learned_infonce | 17 | 81.53 | 82.69 | 84.75 | 83.57 | 83.24 |
| learned_infonce_js | 17 | 79.01 | 82.57 | 84.74 | 83.52 | 82.69 |
| chain_aware_fixed_m | 17 | 85.79 | 85.46 | 87.50 | 86.11 | 86.24 |

### Task-IL — Noise per corruption

| Run | Seed | Gauss | Shot | Impulse |
| --- | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 80.77 | 82.84 | 79.01 |
| feature_mse_seed17 | 17 | 81.26 | 83.40 | 78.58 |
| feature_mse_seed17_lambda0 | 17 | 79.19 | 81.86 | 76.78 |
| feature_mse_seed18 | 18 | 85.26 | 86.61 | 82.81 |
| feature_mse_seed18_lambda0 | 18 | 81.88 | 83.66 | 79.52 |
| feature_mse_seed19 | 19 | 82.44 | 85.02 | 79.56 |
| feature_mse_seed19_lambda0 | 19 | 79.59 | 82.85 | 76.64 |
| infonce_seed17 | 17 | 84.30 | 85.99 | 83.33 |
| classaware_infonce_seed17 | 17 | 85.74 | 87.24 | 85.06 |
| mse05_nce05_seed17 | 17 | 83.42 | 85.36 | 82.28 |
| mse02_nce1_seed17 | 17 | 82.96 | 84.82 | 82.44 |
| augce_seed17 | 17 | 83.49 | 85.25 | 81.31 |
| barlow_seed17 | 17 | 83.46 | 84.98 | 81.63 |
| bt_threepair_seed17 | 17 | 80.82 | 82.78 | 80.17 |
| multihead_seed17 | 17 | 82.46 | 83.91 | 80.89 |
| infonce_recon_seed17 | 17 | 85.12 | 86.82 | 83.86 |
| learned_js | 17 | 81.52 | 83.01 | 78.81 |
| random_js | 17 | 83.13 | 84.93 | 81.60 |
| random_infonce | 17 | 86.16 | 87.54 | 85.00 |
| learned_infonce | 17 | 81.41 | 83.45 | 79.73 |
| learned_infonce_js | 17 | 78.43 | 81.49 | 77.11 |
| chain_aware_fixed_m | 17 | 85.58 | 87.19 | 84.60 |

### Task-IL — Blur per corruption

| Run | Seed | Defocus | Glass | Motion | Zoom |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 87.97 | 79.30 | 85.00 | 87.31 |
| feature_mse_seed17 | 17 | 86.83 | 78.25 | 84.99 | 86.11 |
| feature_mse_seed17_lambda0 | 17 | 84.29 | 76.24 | 82.51 | 82.57 |
| feature_mse_seed18 | 18 | 88.54 | 81.53 | 86.01 | 86.95 |
| feature_mse_seed18_lambda0 | 18 | 85.11 | 78.54 | 82.61 | 83.02 |
| feature_mse_seed19 | 19 | 87.16 | 77.87 | 84.91 | 86.01 |
| feature_mse_seed19_lambda0 | 19 | 84.47 | 75.93 | 82.40 | 82.65 |
| infonce_seed17 | 17 | 87.87 | 83.58 | 86.42 | 87.55 |
| classaware_infonce_seed17 | 17 | 90.00 | 81.65 | 87.96 | 89.49 |
| mse05_nce05_seed17 | 17 | 89.26 | 81.19 | 86.33 | 88.93 |
| mse02_nce1_seed17 | 17 | 90.02 | 83.29 | 88.63 | 89.87 |
| augce_seed17 | 17 | 88.94 | 80.90 | 87.14 | 88.12 |
| barlow_seed17 | 17 | 88.15 | 79.22 | 86.25 | 87.84 |
| bt_threepair_seed17 | 17 | 87.28 | 80.05 | 83.78 | 86.09 |
| multihead_seed17 | 17 | 85.73 | 80.12 | 84.85 | 85.58 |
| infonce_recon_seed17 | 17 | 89.30 | 83.46 | 87.40 | 88.78 |
| learned_js | 17 | 85.67 | 77.14 | 83.40 | 84.82 |
| random_js | 17 | 88.88 | 79.21 | 86.49 | 87.83 |
| random_infonce | 17 | 89.74 | 82.44 | 87.04 | 88.90 |
| learned_infonce | 17 | 86.75 | 76.52 | 82.54 | 84.96 |
| learned_infonce_js | 17 | 85.85 | 77.53 | 82.29 | 84.61 |
| chain_aware_fixed_m | 17 | 88.35 | 80.99 | 85.39 | 87.10 |

### Task-IL — Weather per corruption

| Run | Seed | Snow | Frost | Fog | Bright |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 85.09 | 84.52 | 84.66 | 88.90 |
| feature_mse_seed17 | 17 | 82.68 | 83.94 | 85.19 | 87.81 |
| feature_mse_seed17_lambda0 | 17 | 82.78 | 83.61 | 84.42 | 88.11 |
| feature_mse_seed18 | 18 | 86.14 | 85.51 | 85.48 | 88.84 |
| feature_mse_seed18_lambda0 | 18 | 83.62 | 82.60 | 84.19 | 87.56 |
| feature_mse_seed19 | 19 | 85.00 | 84.61 | 84.90 | 88.72 |
| feature_mse_seed19_lambda0 | 19 | 84.45 | 83.38 | 84.02 | 88.86 |
| infonce_seed17 | 17 | 85.03 | 85.24 | 84.09 | 88.16 |
| classaware_infonce_seed17 | 17 | 86.35 | 85.88 | 86.90 | 90.00 |
| mse05_nce05_seed17 | 17 | 84.99 | 85.20 | 85.81 | 89.16 |
| mse02_nce1_seed17 | 17 | 85.26 | 85.93 | 87.04 | 89.84 |
| augce_seed17 | 17 | 84.50 | 84.67 | 86.97 | 88.68 |
| barlow_seed17 | 17 | 84.08 | 84.25 | 85.02 | 88.03 |
| bt_threepair_seed17 | 17 | 83.70 | 83.70 | 84.14 | 88.27 |
| multihead_seed17 | 17 | 82.33 | 83.87 | 84.28 | 86.81 |
| infonce_recon_seed17 | 17 | 85.74 | 86.13 | 86.24 | 89.94 |
| learned_js | 17 | 83.73 | 83.62 | 84.82 | 88.09 |
| random_js | 17 | 85.08 | 85.03 | 86.48 | 89.18 |
| random_infonce | 17 | 87.39 | 87.00 | 87.96 | 90.58 |
| learned_infonce | 17 | 84.66 | 82.86 | 83.22 | 88.26 |
| learned_infonce_js | 17 | 83.21 | 82.99 | 84.71 | 88.06 |
| chain_aware_fixed_m | 17 | 86.67 | 86.35 | 86.87 | 90.10 |

### Task-IL — Digital per corruption

| Run | Seed | Contrast | Elastic | Pixel | JPEG |
| --- | ---: | ---: | ---: | ---: | ---: |
| metamix_seed17 | 17 | 76.60 | 87.33 | 86.75 | 87.13 |
| feature_mse_seed17 | 17 | 78.91 | 85.84 | 85.95 | 86.71 |
| feature_mse_seed17_lambda0 | 17 | 76.82 | 84.51 | 84.83 | 86.57 |
| feature_mse_seed18 | 18 | 79.09 | 87.09 | 87.60 | 88.75 |
| feature_mse_seed18_lambda0 | 18 | 77.34 | 85.30 | 83.71 | 86.64 |
| feature_mse_seed19 | 19 | 77.77 | 86.25 | 85.28 | 87.09 |
| feature_mse_seed19_lambda0 | 19 | 78.13 | 85.59 | 84.70 | 87.09 |
| infonce_seed17 | 17 | 75.85 | 87.30 | 87.75 | 87.11 |
| classaware_infonce_seed17 | 17 | 78.96 | 88.93 | 89.66 | 88.98 |
| mse05_nce05_seed17 | 17 | 77.20 | 87.88 | 88.36 | 88.18 |
| mse02_nce1_seed17 | 17 | 81.33 | 88.55 | 88.04 | 87.21 |
| augce_seed17 | 17 | 80.53 | 87.38 | 87.03 | 87.74 |
| barlow_seed17 | 17 | 77.38 | 86.67 | 86.97 | 87.09 |
| bt_threepair_seed17 | 17 | 77.72 | 86.22 | 86.94 | 87.38 |
| multihead_seed17 | 17 | 79.81 | 84.82 | 84.69 | 84.54 |
| infonce_recon_seed17 | 17 | 80.42 | 88.31 | 88.63 | 88.19 |
| learned_js | 17 | 76.38 | 85.66 | 86.27 | 87.08 |
| random_js | 17 | 80.73 | 87.56 | 87.10 | 87.98 |
| random_infonce | 17 | 80.94 | 88.42 | 89.19 | 89.59 |
| learned_infonce | 17 | 72.64 | 86.62 | 87.01 | 88.02 |
| learned_infonce_js | 17 | 76.47 | 85.68 | 85.33 | 86.60 |
| chain_aware_fixed_m | 17 | 79.25 | 87.43 | 88.96 | 88.79 |
