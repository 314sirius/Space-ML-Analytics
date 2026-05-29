## 📊 Dataset & Feature Engineering

* **Source:** [NASA Exoplanet Archive - Kepler Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

The dataset tracks **9,564 observation entries** across **49 astronomical columns**. The engine parses features split into distinct functional groups:

* **Transit (Planetary) Variables:** `koi_period` (Orbital Period), `koi_duration` (Transit Duration), `koi_depth` (Transit Depth), `koi_prad` (Planetary Radius), `koi_insol` (Insolation Flux).
* **Stellar Characteristics:** `koi_steff` (Stellar Effective Temperature), `koi_srad` (Stellar Radius), `koi_slogg` (Surface Gravity).
* **Target Label Configuration:** Binomial target filtering out `CANDIDATE` states to focus purely on high-confidence `CONFIRMED` planets versus established `FALSE POSITIVE` detections.

---

## 🛠️ Tech Stack & Methodology

* **Data Processing & Engineering:** `Pandas`, `NumPy`, `MinMaxScaler`
* **Analytical Formulations:** `SymPy` (Symbolic mathematical operations and timing analyses)
* **Machine Learning Models:**
  * **L1-Regularized Logistic Regression** (Sparse baseline selection)
  * **Decision Tree Classifier** (Capturing initial split conditions)
  * **Random Forest Ensemble** (Aggregated forest to minimize variance and combat overfitting)
* **Validation Metrics:** ROC-AUC Curves, Precision, Recall/Sensitivity, Specificity, Confusion Matrices, and **Partial Dependence Plots (PDP)** for interpretability.

---

## 📉 Model Performance & Comparative Summary

| ML Architecture | Area Under Curve (ROC-AUC) | Overall Accuracy | Primary Drivers Detected |
| :--- | :---: | :---: | :--- |
| **Random Forest (Ensemble)** | **0.94** | **0.86** | `log_koi_prad`, `log_koi_period`, `log_koi_tce_plnt_num` |
| **Decision Tree** | **0.91** | **0.865** | `log_koi_period`, `log_koi_prad`, `log_koi_impact` |
| **L1 Logistic Regression** | **0.84** | — | `log_koi_prad`, `log_koi_period` |

### 🔑 Key Analytics Insights

1. **The Non-Linear Reality:** Tree-based architectures significantly out-performed Logistic Regression. The dataset exhibits strict non-linear thresholds (e.g., specific planet radius cutoffs) that linear models inherently fail to fully grasp.
2. **The `log_koi_prad` Filter:** Partial Dependence Plots reveal that planetary size acts as a strict gateway. For `log_koi_prad` values $< 0.35$, probability scores for exoplanet validation are maximized. Beyond $0.4$, the predicted probability drops flatly to zero.
3. **Orbital Window:** The Random Forest successfully isolated an optimal orbital period window ($0.3$ to $0.7$ on log scale), filtering out coordinates noise (`ra`, `dec`) and purely emphasizing the physical indicators of a solar system structure.
