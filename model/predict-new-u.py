import json
import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice
from collections import Counter
import joblib

def calculate_local_features(data):
    """
    Calculate local features for a new material.
    
    Parameters:
        data (dict): Structural data with positions_fractional, geometry, species, etc.
    
    Returns:
        list: List of dictionaries with local features for each atom.
    """
    try:
        species = data.get("species", [])
        if isinstance(species, str):
            species = species.split(',') if species else []
        
        geometry = data.get("geometry", [])
        if isinstance(geometry, str):
            geometry = [float(p) for p in geometry.split(',')] if geometry else []
        elif isinstance(geometry, list):
            geometry = [float(p) for p in geometry]
        
        positions = data.get("positions_fractional", [])
        if isinstance(positions, str):
            positions = [
                [float(p) for p in pos_set.split(',')] 
                for pos_set in positions.split(';') if pos_set
            ]
        elif isinstance(positions, list):
            positions = [
                [float(p) for p in pos_set.split(',')] if isinstance(pos_set, str) else [float(p) for p in pos_set]
                for pos_set in positions if pos_set
            ]
        
        natoms = data.get("natoms", len(positions))
        if isinstance(natoms, str):
            natoms = int(float(natoms))
        
        if not (species and geometry and positions and len(geometry) >= 6 and len(positions) > 0):
            print(f"Error: Incomplete structural data for {data.get('compound', 'Unknown')}")
            return []

        if data.get('spacegroup', '').startswith(('Fm-3c', 'F-43m')):
            expected_a = 5.5 if 'O4' in data.get('compound', '') else 4.8
            if abs(geometry[0] - expected_a) > 1.0 or geometry[0] != geometry[1] or geometry[1] != geometry[2]:
                print(f"Warning: Geometry mismatch for {data.get('compound')}: Expected a~{expected_a}, got {geometry}")

        stoichiometry = data.get("stoichiometry", [])
        if isinstance(stoichiometry, str):
            stoichiometry = [float(s) for s in stoichiometry.split(',')] if stoichiometry else []
        elif isinstance(stoichiometry, list):
            stoichiometry = [float(s) for s in stoichiometry]
        
        total_atoms = natoms if natoms > 0 else len(positions)
        if stoichiometry and len(stoichiometry) == len(species):
            atom_counts = [int(round(s * sum(stoichiometry) * total_atoms)) for s in stoichiometry]
            if sum(atom_counts) != total_atoms:
                print(f"Warning: Stoichiometry mismatch for {data.get('compound')}: "
                      f"atom_counts={atom_counts}, total_atoms={total_atoms}")
            expanded_species = []
            for elem, count in zip(species, atom_counts):
                expanded_species.extend([elem] * count)
        else:
            expanded_species = species if len(species) == len(positions) else []

        if len(expanded_species) != len(positions):
            print(f"Error: Species expansion failed for {data.get('compound')}")
            return []

        lattice = Lattice.from_parameters(
            a=geometry[0], b=geometry[1], c=geometry[2],
            alpha=geometry[3], beta=geometry[4], gamma=geometry[5]
        )
        structure = Structure(lattice, expanded_species, positions, coords_are_cartesian=False)
        
        bond_ranges = {
            'Cu-O': (1.8, 2.6), 'Ti-O': (1.8, 2.5), 'Ir-O': (1.9, 2.5), 'Ag-O': (2.3, 2.9),
            'Os-O': (1.8, 2.2), 'Rb-O': (2.7, 3.2), 'Na-O': (2.2, 2.7), 'K-O': (2.5, 3.0),
            'Cl-O': (1.4, 1.6), 'Sb-O': (1.9, 2.3), 'Mo-O': (1.8, 2.2), 'O-O': (2.3, 2.8),
            'Tb-O': (2.2, 2.8)  
        }
        max_cutoff = 4.5
        
        local_features = []
        
        for idx, site in enumerate(structure):
            neighbors = structure.get_all_neighbors(r=max_cutoff, include_index=True)
            site_neighbors = [n for n in neighbors[idx] if n[1] > 0.1]
            relevant_neighbors = []
            if site.species_string == 'O':
                for n in site_neighbors:
                    pair = f"{n.species_string}-O" if n.species_string != 'O' else 'O-O'
                    min_dist, max_dist = bond_ranges.get(pair, (1.4, max_cutoff))
                    if min_dist <= n[1] <= max_dist:
                        relevant_neighbors.append(n)
            else:
                for n in site_neighbors:
                    if n.species_string == 'O':
                        pair = f"{site.species_string}-O"
                        min_dist, max_dist = bond_ranges.get(pair, (1.4, max_cutoff))
                        if min_dist <= n[1] <= max_dist:
                            relevant_neighbors.append(n)
            
            coordination_number = len(relevant_neighbors)
            bond_lengths = [n[1] for n in relevant_neighbors]
            average_bond_length = np.mean(bond_lengths) if bond_lengths else 0
            neighbor_types = [n.species_string for n in relevant_neighbors]
            neighbor_type_counts = Counter(neighbor_types)
            oxygen_fraction = neighbor_type_counts.get('O', 0) / coordination_number if coordination_number > 0 else 0

            feature = {
                'element': site.species_string,
                'coordination_number': coordination_number,
                'average_bond_length': average_bond_length,
                'neighbor_type_counts': dict(neighbor_type_counts),
                'oxygen_fraction': oxygen_fraction
            }
            local_features.append(feature)

            if coordination_number == 0:
                print(f"Warning: Zero coordination for {data.get('compound')}, {site.species_string}")
                print(f"  All neighbors: {[(n.species_string, n[1]) for n in site_neighbors]}")

        return local_features
    except Exception as e:
        print(f"Error calculating local features for {data.get('compound', 'Unknown')}: {e}")
        return []

def load_new_material_data(input_dir="new_materials"):
    """
    Load new material data and extract features.
    
    Parameters:
        input_dir (str): Directory with JSON files for new materials.
    
    Returns:
        pd.DataFrame: DataFrame with features for prediction.
    """
    data_list = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(input_dir, filename), "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                compound = data.get("compound", "Unknown")
                volume_cell = data.get("volume_cell", 0)
                density = data.get("density", 0)
                egap = data.get("egap", 0)
                
                local_features = calculate_local_features(data)
                if not local_features:
                    print(f"Warning: No local features for {compound}")
                    continue
                
                for local in local_features:
                    element = local.get("element")
                    coordination_number = local.get("coordination_number", 0)
                    average_bond_length = local.get("average_bond_length", 0)
                    oxygen_fraction = local.get("oxygen_fraction", 0)
                    neighbor_type_counts = local.get("neighbor_type_counts", {})
                    
                    feature_dict = {
                        "compound": compound,
                        "element": element,
                        "volume_cell": volume_cell,
                        "density": density,
                        "egap": egap,
                        "coordination_number": coordination_number,
                        "average_bond_length": average_bond_length,
                        "oxygen_fraction": oxygen_fraction
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

def predict_hubbard_u(df, model_path="hubbard_model.pkl"):
    """
    Predict Hubbard U values for new materials using the trained model.
    
    Parameters:
        df (pd.DataFrame): DataFrame with features for new materials.
        model_path (str): Path to the trained model.
    
    Returns:
        pd.DataFrame: DataFrame with predicted U values.
    """
    if df.empty:
        print("Error: No valid data loaded")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print("Average bond length stats for new materials:")
    print(df["average_bond_length"].describe())
    print(f"Warning: Suspicious bond lengths (< 1.0 Å):")
    print(df[df["average_bond_length"] < 1.0][["compound", "element", "average_bond_length"]])
    
    df["average_bond_length"] = df["average_bond_length"].clip(upper=3.5)
    
    # Get expected feature columns from the model
    expected_features = model.feature_names_in_
    
    # Define core features
    core_features = [
        "volume_cell", "density", "egap",
        "coordination_number", "average_bond_length", "oxygen_fraction", "badger_net_charge", "badger_atomic_volume"
    ]
    
    # Add neighbor count features from training
    feature_columns = core_features + [col for col in expected_features if col.startswith("neighbor_")]
    
    # Check for unseen features in new data
    new_columns = [col for col in df.columns if col.startswith("neighbor_") and col not in expected_features]
    if new_columns:
        print(f"Warning: Dropping unseen features: {new_columns}")
        df = df.drop(columns=new_columns)
    
    # Add missing expected features with zeros
    missing_cols = [col for col in feature_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    
    X = df[feature_columns]
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    U_pred = model.predict(X)
    
    df_out = df[["compound", "element"]].copy()
    df_out["U_pred"] = U_pred
    
    df_out.to_csv("new_material_predictions.csv", index=False)
    print("Saved predictions to 'new_material_predictions.csv'")
    
    return df_out

def main():
    """
    Main function to predict Hubbard U values for new materials.
    """
    df = load_new_material_data(input_dir="new_materials")
    
    if df.empty:
        print("Error: No valid data loaded")
        return
    
    predictions = predict_hubbard_u(df, model_path="hubbard_model.pkl")
    
    if predictions is not None:
        print("Predictions completed successfully")
        print(predictions[["compound", "element", "U_pred"]])

if __name__ == "__main__":
    main()