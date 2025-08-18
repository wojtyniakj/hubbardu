import json
import os
import pandas as pd

def extract_distinct_compounds(data_dir="data_most_common"):
    """
    Extract distinct compounds from JSON files in the specified directory.
    
    Parameters:
        data_dir (str): Directory containing AFLOW JSON files.
    
    Returns:
        pd.DataFrame: DataFrame with distinct compounds and their counts.
    """
    compounds = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(data_dir, filename), "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                compound = data.get("compound", None)
                if compound:
                    compounds.append(compound)
                else:
                    print(f"Warning: No compound field found in {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    if not compounds:
        print("Error: No compounds found in the dataset")
        return pd.DataFrame()
    
    # Get unique compounds and their counts
    unique_compounds = sorted(set(compounds))
    compound_counts = [compounds.count(c) for c in unique_compounds]
    
    # Create DataFrame
    df = pd.DataFrame({
        'compound': unique_compounds,
        'count': compound_counts
    })
    
    # Print summary
    print(f"Total files processed: {len([f for f in os.listdir(data_dir) if f.endswith('.json')])}")
    print(f"Total distinct compounds: {len(unique_compounds)}")
    print("\nDistinct compounds (with occurrence counts):")
    print(df)
    
    # Save to CSV
    df.to_csv("distinct_compounds.csv", index=False)
    print("Saved distinct compounds to 'distinct_compounds.csv'")
    
    return df

def main():
    """
    Main function to extract distinct compounds.
    """
    df = extract_distinct_compounds(data_dir="data_most_common")
    
    if not df.empty:
        print("Extraction completed successfully")

if __name__ == "__main__":
    main()