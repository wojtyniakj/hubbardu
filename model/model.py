import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from collections import Counter
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def load_aflow_data(data_dir="data"):
    """
    Load AFLOW JSON files and extract features for Hubbard U model.
    """
    data_list = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(data_dir, filename), "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                compound = data.get("compound", "Unknown")
                volume_cell = data.get("volume_cell", 0)
                density = data.get("density", 0)
                egap = data.get("egap", 0)
                
                ldau = data.get("ldau", "")
                if isinstance(ldau, str):
                    u_values = [float(u) for u in ldau.split(",") if u]
                else:
                    u_values = ldau if isinstance(ldau, list) else []
                
                species = data.get("species", [])
                if len(u_values) != len(species):
                    # print(f"Warning: U values mismatch for {compound}: {u_values}, species: {species}")
                    continue
                
                local_features = data.get("local_features", [])
                if not local_features:
                    # print(f"Warning: No local features for {compound}")
                    continue
                
                for idx, local in enumerate(local_features):
                    element = local.get("element")
                    u_value = u_values[species.index(element)] if element in species else 0
                    
                    if u_value <= 0:
                        continue
                    
                    coordination_number = local.get("coordination_number", 0)
                    average_bond_length = local.get("average_bond_length", 0)
                    oxygen_fraction = local.get("oxygen_fraction", 0)
                    neighbor_type_counts = local.get("neighbor_type_counts", {})

                    # --- BADGER FEATURES ---
                    badger_net_charge = coordination_number / max(average_bond_length, 1e-6)
                    n_atoms = len(species)
                    badger_atomic_volume = volume_cell / max(n_atoms, 1)
                    
                    feature_dict = {
                        "compound": compound,
                        "element": element,
                        "U": u_value,
                        "volume_cell": volume_cell,
                        "density": density,
                        "egap": egap,
                        "coordination_number": coordination_number,
                        "average_bond_length": average_bond_length,
                        "oxygen_fraction": oxygen_fraction,
                        "badger_net_charge": badger_net_charge,
                        "badger_atomic_volume": badger_atomic_volume
                    }
                    
                    for neighbor, count in neighbor_type_counts.items():
                        feature_dict[f"neighbor_{neighbor}_count"] = count
                    
                    data_list.append(feature_dict)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    df = pd.DataFrame(data_list)
    neighbor_columns = [col for col in df.columns if col.startswith("neighbor_")]
    df[neighbor_columns] = df[neighbor_columns].fillna(0)
    return df

def train_hubbard_u_model(df):
    """
    Train a Random Forest model, save it, and generate result plots.
    """
    df = df[df["U"] > 0].copy()
    df["average_bond_length"] = df["average_bond_length"].clip(upper=3.5)
    
    if df.empty:
        print("Error: No valid data with non-zero U values")
        return None
    
    feature_columns = [
        "volume_cell", "density", "egap",
        "coordination_number", "average_bond_length", "oxygen_fraction", "badger_net_charge", "badger_atomic_volume"
    ] + [col for col in df.columns if col.startswith("neighbor_")]
    
    X = df[feature_columns]
    y = df["U"]
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Zmieniono na 80/20
    
    model = RandomForestRegressor(n_estimators=5, max_depth=50,random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"MAE: {mae:.3f} eV")
    print(f"R²: {r2:.3f}")
    print("Feature importances:")
    print(feature_importances.head(10))
    
    df_test = X_test.copy()
    df_test["U_true"] = y_test
    df_test["U_pred"] = y_pred
    df_test["compound"] = df.loc[X_test.index, "compound"]
    df_test["element"] = df.loc[X_test.index, "element"]
    df_test.to_csv("predictions.csv", index=False)
    print("Saved predictions to 'predictions.csv'")
    
    joblib.dump(model, "hubbard_model.pkl")
    print("Saved model to 'hubbard_model.pkl'")

    print("\nGenerating improved prediction plot...")

    plt.figure(figsize=(10, 10)) 

    sns.scatterplot(data=df_test, x="U_true", y="U_pred", alpha=0.5, s=40, edgecolor='k', linewidth=0.5)
    

    plt.xlabel("Rzeczywiste U [eV]", fontsize=14)
    plt.ylabel("Przewidywane U [eV]", fontsize=14)
    plt.title("Porównanie Wartości Rzeczywistych i Przewidywanych", fontsize=18, weight='bold')


    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
   
    max_val = max(df_test["U_true"].max(), df_test["U_pred"].max())
    min_val = min(df_test["U_true"].min(), df_test["U_pred"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label="Idealna predykcja")

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    

    plt.savefig("predictions_scatter_plot.png", dpi=300, bbox_inches='tight')
    print("Saved improved plot to 'predictions_scatter_plot.png'")
    

    
    return {
        "model": model,
        "feature_importances": feature_importances,
        "mae": mae,
        "r2": r2
    }

def main():
    """
    Main function to load data and train the Hubbard U model.
    """
    df = load_aflow_data(data_dir="../data-extractor/data")
    
    if df.empty:
        print("Error: No valid data loaded")
        return
    
    result = train_hubbard_u_model(df)
    
    if result:
        print("\nModel training completed successfully.")
    else:
        print("\nModel training failed.")

if __name__ == "__main__":
    main()