"""
This script trains the forward emulator based on the previously MODFLOW 6 runs.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


OUT_WS = Path(r"C:\Users\\sebas\\Documents\\AYUDANTE SNI\\TESIS\AI\\exp_mixed_200_seed200") # where ml_dataset_abs.npz lives 
BASE_WS = Path(r"C:\\Users\\sebas\\Documents\\AYUDANTE SNI\\TESIS\\yucatan_modelD\\quadtreeGrid\\gridgen_disu")

# This following function computes RMSE using sklearns's mean_squared_error. 
# If 'a' and 'b' are (n_samples,135), sklearn computes MSE averaged across outputs (multioutput uniform average), then sqrt -> one overall RMSE.

def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def main():

    # This blocks loads the dataset: ---------------------------------------------------------------
    ds = np.load(OUT_WS / "ml_dataset.npz", allow_pickle=True)
    X = ds["X"].astype(float)        # (N, 10)
    Y = ds["Y"].astype(float)        # (N, 135)
    status = ds["status"].astype(int)

    # Loads: X = parameters, Y = simulated heads, status = success flag
    # ---------------------------------------------------------------------------------------------
    # Keeps only successful MF6 runs. 
    ok = status == 1
    X = X[ok, :]
    Y = Y[ok, :]

    ## GET OBSERVATIONS IDs (for labeling Y columns)
    # Observation order (must match how sim_heads.dat was written)
    sim0 = pd.read_csv(OUT_WS / "run_00000" / "sim_heads.dat", sep=r"\s+", header=None, names=["obs_id", "head_sim"])
    #obs_names = pd.read_csv(BASE_WS / "obs_heads.csv")["obs_id"].astype(str).to_numpy()
    obs_names = sim0["obs_id"].astype(str).to_numpy()

    if Y.shape[1] != len(obs_names):
        raise ValueError(f"Y has {Y.shape[1]} cols but obs_heads.csv has {len(obs_names)} obs ids.")

    ## TRAIN / TEST SPLIT
    # Randomly keeps 20% for testing, fixed seed for reproducibility.
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=7) # Xtr, Ytr: X and Y for training; Xte, Yte: X and Y for testing

    ## STANDARDIZATION (very important for MLP) USING ONLY THE TRAINING SET --------------------------------------------------------------------
    # This is almost mandatory for stable neural net training.

    xsc = StandardScaler().fit(Xtr) ## ----> Scales each parameter to mean 0 / std 1.
    ysc = StandardScaler().fit(Ytr) ## ----> Scales each output head to mean 0 / std 1 across training samples
    ## This standarization helps because:
    # - Avoids that variables with very different scales dominaining the trainment
    # - Stabilize the optimization
    # - Improves Adam convex
    # - The network learns relative patterns and dont get stucked by variables magnitudes 

    Xtr_s = xsc.transform(Xtr) ## Now here i transform train and test datasets with this scalers
    Xte_s = xsc.transform(Xte)
    Ytr_s = ysc.transform(Ytr)
    Yte_s = ysc.transform(Yte)

    # --------------------------------------------------------------------------------------------------------------
    # NEURAL NET EMULATOR:  This trains one neural network net mapping 10 inputs ------> 135 outputs.

    model = MLPRegressor(
        hidden_layer_sizes=(256, 256), # 2 hidden layers with 256 neurons by layer
        activation="relu", 
        solver="adam",
        alpha=1e-4, # L2 Regularization with alpha=1e-4
        learning_rate_init=1e-3, # Starting learning rate
        max_iter=8000, # number of iteration 
        early_stopping=True,
        n_iter_no_change=40, # early-stopping with 40 of patience  
        random_state=7,
        verbose=True
    )
    # So the model is trained:   
    model.fit(Xtr_s, Ytr_s)
    # ---------------------------------------------------------------------------------------------------------------
    # EVALUATE IN REAL HEAD UNITS:
    # -Predicts standardized outputs
    # -Unscale to meters
    # - Reports overall RMSE and per-observation RMSE distribution

    Ypred_s = model.predict(Xte_s) # ---> First i got the standarized prediction 
    Ypred = ysc.inverse_transform(Ypred_s) # ---> Now i return back the prediction in physical units of meters

    overall = rmse(Yte, Ypred)  ## Computing global RMSE 
    per_obs = np.sqrt(np.mean((Yte - Ypred) ** 2, axis=0)) ## Computing RMSE by each observation 

    print(f"\nEmulator test RMSE (overall): {overall:.6f}") 
    print(f"Median per-obs RMSE: {np.median(per_obs):.6f}")
    print(f"95th percentile per-obs RMSE: {np.quantile(per_obs, 0.95):.6f}")
    # ---------------------------------------------------------------------------------------------------------------
    # SAVING EVERYTHING
    pack = {
        "model": model,
        "x_scaler": xsc,
        "y_scaler": ysc,
        "par_names": ds["par_names"],
        "obs_names": obs_names,
    }
    out = OUT_WS / "forward_emulator.joblib"
    dump(pack, out)
    print(f"\nSaved emulator: {out}")


if __name__ == "__main__":
    main()
