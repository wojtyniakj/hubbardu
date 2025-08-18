from collections import Counter
from pymatgen.core import Structure, Lattice
import requests
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from ase import Atoms
from ase.neighborlist import neighbor_list

progress_lock = threading.Lock()  

def calculate_local_features(data):
    """
    Calculate local features from AFLOW JSON data with element-specific bond length cutoffs.
    
    Parameters:
        data (dict): AFLOW JSON data with positions_fractional, geometry, species, and natoms.
    
    Returns:
        list: List of dictionaries with local features for each atom.
    """
    try:
        # Handle species
        species = data.get("species", [])
        if isinstance(species, str):
            species = species.split(',') if species else []
        
        # Handle geometry
        geometry = data.get("geometry", [])
        if isinstance(geometry, str):
            geometry = [float(p) for p in geometry.split(',')] if geometry else []
        elif isinstance(geometry, list):
            geometry = [float(p) for p in geometry]
        
        # Handle positions_fractional
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
        
        # Handle stoichiometry
        stoichiometry = data.get("stoichiometry", [])
        if isinstance(stoichiometry, str):
            stoichiometry = [float(s) for s in stoichiometry.split(',')] if stoichiometry else []
        elif isinstance(stoichiometry, list):
            stoichiometry = [float(s) for s in stoichiometry]
        
        # Handle natoms
        natoms = data.get("natoms", len(positions))
        if isinstance(natoms, str):
            natoms = int(float(natoms))
        



        # Calculate total atoms
        total_atoms = natoms if natoms > 0 else len(positions)
        if stoichiometry and len(stoichiometry) == len(species):
            atom_counts = [int(round(s * sum(stoichiometry) * total_atoms)) for s in stoichiometry]
            if sum(atom_counts) != total_atoms:
                print(f"Warning: Stoichiometry mismatch for {data.get('compound')}: "
                      f"atom_counts={atom_counts}, total_atoms={total_atoms}, stoichiometry={stoichiometry}")
        
        # Validate inputs
        if not (species and geometry and positions and len(geometry) >= 6 and len(positions) > 0):
            print(f"Error: Incomplete structural data for {data.get('compound', 'Unknown')}: "
                  f"species={len(species)}, geometry={len(geometry)}, positions={len(positions)}, "
                  f"total_atoms={total_atoms}")
            return []

        # Validate geometry for FCC (Fm-3c or F-43m)
        expected_a = 5.5 if 'O4' in data.get('compound') else 4.8  # Larger for AgClO4, smaller for AgMO
        if data.get('sg', '').startswith(('Fm-3c', 'F-43m')):
            if abs(geometry[0] - expected_a) > 1.0 or geometry[0] != geometry[1] or geometry[1] != geometry[2] or \
               geometry[3] != 60 or geometry[4] != 60 or geometry[5] != 60:
                print(f"Warning: Geometry mismatch for {data.get('compound')}: Expected a~{expected_a}, α=β=γ=60, got {geometry}")
            # Check volume_cell consistency
            volume_expected = (geometry[0] ** 3) / 4  # FCC unit cell volume (a^3 / 4)
            if abs(data.get('volume_cell', 0) - volume_expected) / volume_expected > 0.2:
                print(f"Warning: Volume mismatch for {data.get('compound')}: Expected volume~{volume_expected}, got {data.get('volume_cell', 0)}")

        # Expand species to match positions
        if len(species) != len(positions):
            atom_counts = [int(round(s * sum(stoichiometry) * total_atoms)) for s in stoichiometry] if stoichiometry else [1] * len(species)
            expanded_species = []
            for elem, count in zip(species, atom_counts):
                expanded_species.extend([elem] * count)
            if len(expanded_species) != len(positions):
                print(f"Error: Species expansion failed for {data.get('compound')}: "
                      f"expanded_species={len(expanded_species)}, positions={len(positions)}")
                return []
        else:
            expanded_species = species

        # Create pymatgen Structure
        lattice = Lattice.from_parameters(
            a=geometry[0], b=geometry[1], c=geometry[2],
            alpha=geometry[3], beta=geometry[4], gamma=geometry[5]
        )
        structure = Structure(lattice, expanded_species, positions, coords_are_cartesian=False)
        
        # Element-specific bond length ranges
        bond_ranges = {
            'Cu-O': (1.8, 2.6),  # Extended for Cu oxides
            'Ti-O': (1.8, 2.5),  # Extended for Ti oxides
            'Ir-O': (1.9, 2.5),  # Extended for Ir oxides
            'Ag-O': (2.3, 2.9),  # Extended for Ag oxides
            'Os-O': (1.8, 2.2),  # OsO6 octahedra
            'Rb-O': (2.7, 3.2),  # Rb+ in oxides
            'Na-O': (2.2, 2.7),  # Na+ in oxides
            'K-O': (2.5, 3.0),   # K+ in oxides
            'Cl-O': (1.4, 1.6),  # ClO4- tetrahedra
            'Sb-O': (1.9, 2.3),  # Sb-O in oxides
            'Mo-O': (1.8, 2.2),  # Mo-O in oxides
            'O-O': (2.3, 2.8)    # O-O in oxides
        }
        max_cutoff = 4.5  # Increased to capture edge cases
        
        local_features = []
        
        for idx, site in enumerate(structure):
            # Get all neighbors within max cutoff
            neighbors = structure.get_all_neighbors(r=max_cutoff, include_index=True)
            site_neighbors = [n for n in neighbors[idx] if n[1] > 0.1]  # Exclude self
            # Filter for relevant bonds
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
                'neighbor_types': neighbor_types,
                'neighbor_type_counts': dict(neighbor_type_counts),
                'oxygen_fraction': oxygen_fraction
            }
            local_features.append(feature)

        return local_features
    except Exception as e:
        print(f"Error calculating local features for {data.get('compound', 'Unknown')}: {e}")
        return []

def get_structure_info(data):
    """
    Parses the AFLOW JSON file to extract key structural information.
    """
    try:
        # Handle species
        species = data.get("species", [])
        if isinstance(species, str):
            species = species.split(',') if species else []

        # Handle geometry
        geometry = data.get("geometry", [])
        if isinstance(geometry, str):
            geometry = [float(p) for p in geometry.split(',')] if geometry else []
        elif isinstance(geometry, list):
            geometry = [float(p) for p in geometry]

        # Handle positions_fractional
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

        # Handle stoichiometry
        stoichiometry = data.get("stoichiometry", [])
        if isinstance(stoichiometry, str):
            stoichiometry = [float(s) for s in stoichiometry.split(',')] if stoichiometry else []
        elif isinstance(stoichiometry, list):
            stoichiometry = [float(s) for s in stoichiometry]

        # Handle natoms
        natoms = data.get("natoms", len(positions))
        if isinstance(natoms, str):
            natoms = int(float(natoms))

        # Handle Bader net charges
        bader_net_charges = data.get("bader_net_charges", [])
        if isinstance(bader_net_charges, str):
            bader_net_charges = [float(v) for v in bader_net_charges.split(',') if v]
        elif isinstance(bader_net_charges, list):
            bader_net_charges = [float(v) for v in bader_net_charges if v not in ("", None)]

        # Handle Bader atomic volumes
        bader_atomic_volumes = data.get("bader_atomic_volumes", [])
        if isinstance(bader_atomic_volumes, str):
            bader_atomic_volumes = [float(v) for v in bader_atomic_volumes.split(',') if v]
        elif isinstance(bader_atomic_volumes, list):
            bader_atomic_volumes = [float(v) for v in bader_atomic_volumes if v not in ("", None)]


        structure_dict = {
            "compound": data.get('compound', 'Unknown'),
            "ldau": data.get("ldau_u", ""),
            "species": species,
            "geometry": geometry,
            "positions_fractional": positions,
            "stoichiometry": stoichiometry,
            "natoms": natoms,
            "volume_cell": float(data.get("volume_cell", 0)),
            "density": float(data.get("density", 0)),
            "spacegroup": data.get("sg", ""),
            "egap": float(data.get("Egap", 0)),
            "spin_cell": float(data.get("spin_cell", 0)),
            "bader_net_charges": bader_net_charges,
            "bader_atomic_volumes": bader_atomic_volumes
        }
        return structure_dict
    except Exception as e:
        print(f"Error parsing structure info for {data.get('compound', 'Unknown')}: {e}")
        return {}

def fetch_aflow_data(uid, structure, output_dir="data", aflow_url="http://aflowlib.duke.edu"):
    """
    Fetch data for a specific UID and structure, calculate local features, and save to JSON.
    """
    url = f"{aflow_url}/AFLOWDATA/{uid}/{structure}?format=json"
    output_file = os.path.join(output_dir, f"aflow_data_{uid.replace('/', '_')}_{structure}.json")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        structure_info = get_structure_info(data)
        if not structure_info.get("ldau") or not structure_info.get("species") or "O" not in structure_info["species"]:
            print(f"Skipping {structure}: No valid ldau or no oxygen")
            return

        local_features = calculate_local_features(structure_info)

        if len(local_features) == len(structure_info.get("bader_net_charges", [])):
            for i, feat in enumerate(local_features):
                feat["bader_net_charge"] = structure_info["bader_net_charges"][i]
                if i < len(structure_info.get("bader_atomic_volumes", [])):
                    feat["bader_atomic_volume"] = structure_info["bader_atomic_volumes"][i]

        merged_data = {
            **structure_info,
            "local_features": local_features
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Saved data for '{structure}' to '{output_file}'")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {uid}/{structure}: {e}")
    except Exception as e:
        print(f"Error processing {uid}/{structure}: {e}")

def get_remaining_structures_count(total_structures, processed_structures_count):
    """
    Calculates and prints the number of remaining structures to process.
    """
    remaining_count = total_structures - processed_structures_count
    print(f"Remaining structures to process: {remaining_count}")

def fetch_aflow_data_batch(aflow_uids, structures, aflow_url="http://aflowlib.duke.edu", max_workers=12):
    """
    Fetch data for multiple UIDs and structures from AFLOWLIB using multithreading.
    """
    total_structures = len(aflow_uids) * len(structures)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fetch_aflow_data, uid, structure, aflow_url=aflow_url): (uid, structure)
            for uid in aflow_uids
            for structure in structures
        }

        for future in as_completed(future_to_task):
            uid, structure = future_to_task[future]
            try:
                future.result()
            except Exception as e:
                with progress_lock:
                    print(f"Task failed for {uid}/{structure}: {e}")
            finally:
                with progress_lock:
                    processed_count += 1
                    get_remaining_structures_count(total_structures, processed_count)

def fetch_and_process_aflow_entries(url, aflow_uid, aflow_url="http://aflowlib.duke.edu"):
    """
    Fetch the list of entries for a given UID and process oxygen-containing structures.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        raw_entries = data.get("aflowlib_entries", {})
        if not isinstance(raw_entries, dict):
            print("Unexpected format in 'aflowlib_entries'")
            return

        structures = list(raw_entries.values())
        print(f"Found {len(structures)} AFLOW structures to process.")

        o_structures = [s for s in structures if "O" in s]
        print(f"Number of structures containing 'O': {len(o_structures)}")
        print(f"Processing {len(o_structures)} structures containing 'O' in parallel...")

        if not o_structures:
            print("No structures containing 'O' found.")
            return

        fetch_aflow_data_batch([aflow_uid], o_structures, aflow_url, max_workers=12)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching entries from {url}: {e}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from response: {url}")
    """
    Fetch the list of entries for a given UID and process oxygen-containing structures.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        raw_entries = data.get("aflowlib_entries", {})
        if not isinstance(raw_entries, dict):
            print("Unexpected format in 'aflowlib_entries'")
            return

        structures = list(raw_entries.values())
        print(f"Found {len(structures)} AFLOW structures to process.")

        o_structures = [s for s in structures if "O" in s]
        print(f"Number of structures containing 'O': {len(o_structures)}")
        print(f"Processing {len(o_structures)} structures containing 'O'.")

        if not o_structures:
            print("No structures containing 'O' found.")
            return

        fetch_aflow_data_batch([aflow_uid], o_structures, aflow_url)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching entries from {url}: {e}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from response: {url}")

if __name__ == "__main__":
    aflow_url = "http://aflowlib.duke.edu"
    aflow_uid = "ICSD_WEB/FCC"
    full_url = f"{aflow_url}/AFLOWDATA/{aflow_uid}/?format=json"
    fetch_and_process_aflow_entries(full_url, aflow_uid, aflow_url)