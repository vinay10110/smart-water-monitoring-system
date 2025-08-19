# Smart Water Monitoring System — Consumption Prediction

A lightweight machine learning project to predict residential water consumption using tabular features (residents, apartment type, weather, pricing, income level, amenities, appliance usage, etc.). This repository includes training data, a trained model artifact, evaluation plots, and a sample submission.

## Project Structure

- `dataset/`
  - `train.csv` — Training data with target `Water_Consumption`
  - `test.csv` — Test data without target
- `results/`
  - `model/SWS.pkl` — Saved trained model (pickle)
  - `plots/` — Evaluation & diagnostics
    - `learning_curve.png`
    - `features_imp.png`
    - `residuals.png`
  - `submission.csv` — Example predictions for the test set
- `SWS.ipynb` — Jupyter notebook for EDA, training, evaluation, and inference
- `requirements.txt` — Project dependencies

## Visualizations

These plots summarize the model’s training behavior and performance:

- Learning Curve

  ![Learning Curve](results/plots/learning_curve.png)

- Feature Importance

  ![Feature Importance](results/plots/features_imp.png)

- Residuals

  ![Residuals](results/plots/residuals.png)

## Setup

- Python 3.x
- Recommended: a virtual environment

```bash
# Create and activate a virtual environment (examples)
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Data

- Train columns (sample):
  - `Timestamp, Residents, Apartment_Type, Temperature, Humidity, Water_Price, Period_Consumption_Index, Income_Level, Guests, Amenities, Appliance_Usage, Water_Consumption`
- Test columns are identical, excluding `Water_Consumption`.

## Reproduce Training / Evaluation

Open the notebook and run cells end-to-end:

```bash
# (Optional) install Jupyter if needed
pip install jupyter

# Launch Jupyter
jupyter notebook SWS.ipynb
```

The notebook will:
- Load data from `dataset/`
- Train a regression model (e.g., scikit-learn / PyCaret pipeline)
- Generate evaluation plots to `results/plots/`
- Save the trained model to `results/model/SWS.pkl`
- Optionally create predictions (e.g., `results/submission.csv`)

## Notes

- Ensure your local environment has compatible versions for libraries in `requirements.txt`.
- If the model was saved using a higher-level framework (e.g., PyCaret), make sure the same package is installed to load the pickle successfully.
- The dataset may contain missing or categorical values; your preprocessing must mirror what was used during training.

## License

Specify a license here if applicable.
