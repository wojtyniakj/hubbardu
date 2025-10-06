
import json
import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_species(data_dir="data"):
    """
    Load species from all JSON files in the data directory.
    
    Parameters:
        data_dir (str): Directory containing AFLOW JSON files.
    
    Returns:
        list: List of all species (excluding oxygen) across compounds.
    """
    all_species = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(data_dir, filename), "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                compound = data.get("compound", "Unknown")
                species = data.get("species", [])
                if isinstance(species, str):
                    species = species.split(',') if species else []
                
                # Exclude oxygen and add unique species for this compound
                unique_species = set(species) - {'O'}
                all_species.extend(unique_species)
                
                if not unique_species:
                    print(f"Warning: No non-oxygen species found in {compound}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return all_species

def analyze_species(data_dir="data"):
    """
    Analyze the frequency of species (excluding oxygen) and print the most common ones.
    
    Parameters:
        data_dir (str): Directory containing AFLOW JSON files.
    
    Returns:
        pd.DataFrame: DataFrame with species counts, sorted by frequency.
    """
    # Load all species
    all_species = load_species(data_dir)
    
    if not all_species:
        print("Error: No species found in the dataset")
        return pd.DataFrame()
    
    # Count frequency of each species
    species_counts = Counter(all_species)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame.from_dict(species_counts, orient='index', columns=['count'])
    df = df.sort_values(by='count', ascending=False).reset_index()
    df.columns = ['species', 'count']
    
    # Print results
    print(f"Total compounds processed: {len([f for f in os.listdir(data_dir) if f.endswith('.json')])}")
    print(f"Total unique non-oxygen species: {len(species_counts)}")
    print("\nMost common species (excluding oxygen):")
    print(df)
    
    return df

def plot_species_counts(df, top_n=10):
    """
    Create a bar chart of the top N most common species.
    
    Parameters:
        df (pd.DataFrame): DataFrame with species and their counts.
        top_n (int): Number of top species to plot.
    
    Returns:
        Chart: Bar chart of species counts.
    """
    if df.empty:
        print("Error: No data to plot")
        return
    
    # Select top N species
    df_plot = df.head(top_n)
    species = df_plot['species'].tolist()
    counts = df_plot['count'].tolist()
    
    plt.figure(figsize=(12, 8))
    plt.bar(species, counts, color='skyblue')
    plt.xlabel('Pierwiastek', fontsize=12)
    plt.ylabel('Liczba wystąpień w związkach', fontsize=12)
    plt.title(f'Top {top_n} najczęściej występujących pierwiastków (poza tlenem)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("species_distribution.png")
    print("\nSaved species distribution plot to 'species_distribution.png'")


def main():
    """
    Main function to analyze species and plot results.
    """
    df = analyze_species(data_dir="data")
    
    if not df.empty:
        plot_species_counts(df, top_n=10)
        df.to_csv("species_counts.csv", index=False)
        print("Saved species counts to 'species_counts.csv'")

if __name__ == "__main__":
    main()