# CS4990 Wildfire Spread Prediction

AI for Wildfire Spread Prediction: Generate predictive wildfire spread maps based on current weather and vegetation conditions using a Wasserstein GAN (wGAN) trained on satellite imagery and vegetation data.

---

## Dataset and Preprocessing

### GIS Fire Map
Satellite fire detection data was overlaid onto a GIS map to visualize active fire pixel locations geographically across the west coast.

<img width="776" height="914" alt="gis_fire_image" src="https://github.com/user-attachments/assets/4658d023-0b38-424c-8b29-29fb87197f28" />

### Rasterized Fire Imagery
Raw satellite imagery was converted into binary rasters distinguishing fire pixels from non-fire pixels, forming the ground truth labels used for model training.

<img width="476" height="736" alt="converted_fire_image" src="https://github.com/user-attachments/assets/922f7377-c553-4de6-a3c5-2660cd91da4c" />

### Vegetation (NDVI) Satellite Imagery
The model uses NDVI (Normalized Difference Vegetation Index) satellite imagery as input, which captures vegetation density and health across the landscape.

<img width="480" height="742" alt="vegetation_image" src="https://github.com/user-attachments/assets/2489b49a-b6c9-4d12-9900-e331da2985f6" />

---

## Model Training

### Early Training Challenges
Because fire pixels are rare relative to non-fire pixels, the model initially struggled to learn the fire dataset and defaulted to learning primarily from the vegetation data. This caused the model to predict that the entire west coast was on fire, as shown below.

<img width="740" height="916" alt="poor_model_prediction" src="https://github.com/user-attachments/assets/f9f511e4-1aa2-474d-ae45-649ebc30735d" />

This class imbalance issue was addressed through fire pixel balancing, curriculum learning, and tuning the fire loss weight during training.

---

## Results

### Model Performance
The final trained model achieved high accuracy and performed well at predicting non-fire pixels. However due to the rarity of fire events in the dataset, the model has a tendency toward false positives and struggles to pinpoint the exact location of fire spread. As a result the model has lower precision and IoU scores, which would likely improve with a larger and more balanced training dataset.

<img width="628" height="442" alt="model_metrics_2" src="https://github.com/user-attachments/assets/138b0f02-95c5-44af-b386-ff8eed6757bf" />

### Final Model Losses
| Metric | Value |
|--------|-------|
| Critic Loss | -0.98 |
| Generator Loss | -4.59 |
| Gradient Penalty (GP) Loss | 0.004 |

### Prediction vs Ground Truth
<img width="1724" height="588" alt="prediction_v_ground_truth_2" src="https://github.com/user-attachments/assets/4a0a23bf-a97b-4b41-ba39-f87a2dea7230" />

---

## Running and Testing the wGAN

### Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy 1.20+
- SciPy 1.7+
- Matplotlib 3.5+
- pandas 1.3+
- scikit-learn 1.0+
- tqdm

### Training
Run `main.py` with your desired arguments. The following arguments were used to train the model described in the project paper:
```bash
--burn_dir ../data/burn_wc_tensors \
--ndvi_dir ../data/ndvi_west_coast_tensors \
--balance_fire \
--curriculum_learning \
--curriculum_start 10 \
--curriculum_end 50 \
--curriculum_max_ndvi 0.8 \
--curriculum_type cosine \
--fire_weight 400.0 \
--lambda_fire 100.0 \
--epochs 100 \
--validate_every 5 \
--loss_plot_every 10 \
--lambda_L1 150
```

### Testing
Once training is complete, evaluate the model using `test_wildfire_gan.py`. The following arguments were used to produce the results in the paper:
```bash
--checkpoint ./new_model/wgan_improved_run2/curriculum_cosine/wgan_ndvi_improved_checkpoint_epoch_100.pt \
--ndvi_dir ../data/2020_data/2020_NDVI_wc_tensors \
--burn_dir ../data/2020_data/2020_burn_wc_tensors \
--ndvi_tif_dir ../data/2020_data/2020_NDVI_wc_tif \
--start_day 230 --end_day 270 --threshold .05
```

The testing program will output visualizations and a summary of metrics for the testing run.

