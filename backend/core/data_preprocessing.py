import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from CSV file.
    """
    return pd.read_csv(filepath)

def drop_unused_columns(df):
    """
    Drop columns that are not used in training.
    """
    # Columns not present in train.py feature lists
    cols_to_drop = ["Posted On", "Point of Contact"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_cols)

def clean_size(df):
    """
    Clean the Size column if necessary.
    """
    # Simply ensure it's integer
    if 'Size' in df.columns:
        df['Size'] = pd.to_numeric(df['Size'], errors='coerce').fillna(0).astype(int)
    return df

def reduce_locality_cardinality(df, threshold=10):
    """
    Group rare Area Locality categories into 'Other'.
    """
    if 'Area Locality' in df.columns:
        counts = df['Area Locality'].value_counts()
        others = counts[counts < threshold].index
        df['Area Locality'] = df['Area Locality'].apply(lambda x: 'Other' if x in others else x)
    return df

def parse_floor(df):
    """
    Parse 'Floor' column into 'current_floor' and 'total_floors'.
    Example: "2 out of 5" -> current=2, total=5
    """
    if 'Floor' not in df.columns:
        return df
        
    def extract_floors(x):
        try:
            if pd.isna(x):
                return 0, 0
            
            parts = str(x).split(' out of ')
            current_str = parts[0].strip().lower()
            
            # Determine current floor
            if 'ground' in current_str:
                current = 0
            elif 'upper basement' in current_str:
                current = -1
            elif 'lower basement' in current_str:
                current = -2
            elif 'basement' in current_str:
                current = -1
            else:
                try:
                    current = int(current_str)
                except ValueError:
                    current = 0 # Fallback
            
            # Determine total floors
            if len(parts) > 1:
                try:
                    total = int(parts[1].strip())
                except ValueError:
                    total = current 
            else:
                total = current 
                
            return current, total
        except Exception:
            return 0, 0

    floors = df['Floor'].apply(extract_floors)
    df['current_floor'] = floors.apply(lambda x: x[0])
    df['total_floors'] = floors.apply(lambda x: x[1])
    
    # Drop original Floor column as it's replaced
    df = df.drop(columns=['Floor'])
        
    return df

def log_transform_target(df):
    """
    Log-transform the target variable 'Rent'.
    """
    if 'Rent' in df.columns:
        df['Rent'] = np.log1p(df['Rent'])
    return df
