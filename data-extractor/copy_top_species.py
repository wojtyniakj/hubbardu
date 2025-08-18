import json
import os
import shutil
from collections import Counter
import pandas as pd

def load_species(data_dir="data"):
    """
    Load species from all JSON files in the data directory.
    
    Parameters:
        data_dir (str): Directory containing AFLOW JSON files.
    
    Returns:
        tuple: (List of all non-oxygen species, dict mapping filenames to species)
    """
    all_species = []
    file_species_map = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(data_dir, filename), "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                compound = data.get("compound", "Unknown")
                species = data.get("species", [])
                if isinstance(species, str):
                    species = species.split(',') if species else []
                
                # Exclude oxygen and store unique species for this compound
                unique_species = set(species) - {'O'}
                if unique_species:
                    all_species.extend(unique_species)
                    file_species_map[filename] = unique_species
                else:
                    print(f"Warning: No non-oxygen species found in {compound} ({filename})")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return all_species, file_species_map

def get_top_species(all_species, top_n=5):
    """
    Get the top N most common species (excluding oxygen).
    
    Parameters:
        all_species (list): List of all non-oxygen species.
        top_n (int): Number of top species to return.
    
    Returns:
        pd.DataFrame: DataFrame with top N species and their counts.
    """
    species_counts = Counter(all_species)
    df = pd.DataFrame.from_dict(species_counts, orient='index', columns=['count'])
    df = df.sort_values(by='count', ascending=False).reset_index()
    df.columns = ['species', 'count']
    return df.head(top_n)

def copy_top_species_files(data_dir="data", output_dir="data_most_common", top_n=5):
    """
    Copy JSON files containing the top N most common non-oxygen species to output_dir.
    
    Parameters:
        data_dir (str): Directory containing AFLOW JSON files.
        output_dir (str): Directory to copy matching JSON files.
        top_n (int): Number of top species to consider.
    
    Returns:
        pd.DataFrame: DataFrame with top species counts.
    """
    # Load species and file mapping
    all_species, file_species_map = load_species(data_dir)
    
    if not all_species:
        print("Error: No species found in the dataset")
        return pd.DataFrame()
    
    # Get top N species
    top_species_df = get_top_species(all_species, top_n)
    top_species = set(top_species_df['species'].tolist())
    
    if not top_species:
        print("Error: No top species identified")
        return pd.DataFrame()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy files containing top species
    copied_files = []
    for filename, species in file_species_map.items():
        if species & top_species:  # Check if any top species are present
            src_path = os.path.join(data_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append(filename)
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    # Print summary
    print(f"Total compounds processed: {len(file_species_map)}")
    print(f"Total unique non-oxygen species: {len(set(all_species))}")
    print(f"Top {top_n} most common species (excluding oxygen):")
    print(top_species_df)
    print(f"\nCopied {len(copied_files)} files to '{output_dir}'")
    if copied_files:
        print("Copied files:")
        for f in copied_files:
            print(f"  {f}")
    
    # Save top species counts
    top_species_df.to_csv("top_species_counts.csv", index=False)
    print("Saved top species counts to 'top_species_counts.csv'")
    
    return top_species_df

def main():
    """
    Main function to copy JSON files with top 5 non-oxygen species.
    """
    df = copy_top_species_files(data_dir="data", output_dir="data_most_common", top_n=5)
    
    if not df.empty:
        print("Analysis and file copying completed successfully")

if __name__ == "__main__":
    main()