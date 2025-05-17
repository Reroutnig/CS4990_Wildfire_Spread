# CS4990_Wildfire_Spread
AI for Wildfire Spread Prediction: Generate predictive wildfire spread maps based on current weather and vegetation conditions

# Running and testing the wGAN
1. Requirements 
    - Python: Version 3.8+
    - PyTorch: Version 1.10+
    - NumPy: Version 1.20+
    - SciPy: Version 1.7+
    - Matplotlib: Version 3.5+
    - pandas: Version 1.3+
    - scikit-learn: Version 1.0+
    - tqdm: For progress tracking

2. Run the program main.py with desired arguments 
    - Here is a list of arguments we used for our model discussed in the project paper:
```bash
    --burn_dir ../data/burn_wc_tensors \
    --ndvi_dir .../data/ndvi_west_coast_tensors \
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
3. Once you are finished training, test the model using the test_wildfire_gan.py program with desired arguments
    - Here is a list of arguments we used to obtain the results discussed in our paper:

```bash
    --checkpoint ./new_model/wgan_improved_run2/curriculum_cosine/wgan_ndvi_improved_checkpoint_epoch_100.pt \
    --ndvi_dir ../data/2020_data/2020_NDVI_wc_tensors \
    --burn_dir ../data/2020_data/2020_burn_wc_tensors \
    --ndvi_tif_dir ../data/2020_data/2020_NDVI_wc_tif \
    --start_day 230 --end_day 270 --threshold .05
```
- The testing program will output visualizations and a summary of the metrics for the testing run.

