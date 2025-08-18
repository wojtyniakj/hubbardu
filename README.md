# Hubbard U ML

# Hubbard U Prediction Model

This project provides a machine learning workflow to predict **Hubbard U parameters** for materials based on AFLOW dataset JSON files. It leverages **Random Forest regression** 

---

## Features

The model uses the following features:

- **Global structure features:**
  - `volume_cell` – Volume of the unit cell
  - `density` – Density of the material
  - `egap` – Band gap

- **Local atomic features:**
  - `coordination_number` – Number of nearest neighbors
  - `average_bond_length` – Average bond length of the atom
  - `oxygen_fraction` – Fraction of oxygen neighbors
  - `neighbor_{element}_count` – Counts of different neighboring elements

- **Badger features:**
  - `badger_net_charge` – Approximate net charge derived from coordination and bond length
  - `badger_atomic_volume` – Approximate atomic volume derived from cell volume

---

## Requirements

```bash
Python >= 3.8
pandas
numpy
scikit-learn
joblib
```

1. Prepare AFLOW JSON data in the data-extractor/data/ folder. 
    - **python3 data-extractor/fetch_aflow_data_threaded.py**
2. Run the training script:
    - **python3 model.py**
3. Output:

    - **predictions.csv**: Test set predictions including true vs predicted U values. 
    - **hubbard_model.pkl**: Trained Random Forest model saved for reuse.
4. Predict values for new structure
    - **python3 predict-new-u.py**, make sure to provide correct json file in new_materials directory(just like in the example for FE1Mo1O6Sr2)

```mermaid 

    flowchart LR
        A[AFLOW JSON files] --> B[Load Data & Extract Features]
        B --> C[Global Features]
        B --> D[Local Atomic Features]
        B --> E[Badger Features]
        C --> F[Feature Matrix]
        D --> F
        E --> F
        F --> G[Random Forest Model Training]
        G --> H[Predicted Hubbard U Values]
        H --> I[Save Predictions & Model]

