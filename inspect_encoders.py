
import pickle
import numpy as np

def inspect_encoders(encoder_path):
    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Encoder file not found at {encoder_path}")
        return

    print("Inspecting encoders.pkl...")

    for key, value in encoders.items():
        print(f"\n--- Field: {key} ---")
        if isinstance(value, dict):
            print(f"  Type: dict")
            num_items = len(value)
            print(f"  Cardinality / Number of items: {num_items}")
            if num_items == 0:
                print("  !! WARNING: Empty dictionary.")
            else:
                print("  Top 5 items:")
                for i, (k, v) in enumerate(value.items()):
                    if i >= 5:
                        break
                    print(f"    {k}: {v}")
        elif isinstance(value, list):
            print(f"  Type: list")
            num_items = len(value)
            print(f"  Cardinality / Length: {num_items}")
            if num_items == 0:
                print("  !! WARNING: Empty list.")
            else:
                print("  Top 5 items:")
                for i, item in enumerate(value):
                    if i >= 5:
                        break
                    print(f"    {item}")
        elif hasattr(value, 'shape'):
            print(f"  Type: numpy array or similar")
            print(f"  Shape: {value.shape}")
        else:
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")

    # Specific checks from prompt
    if 'merchant_cluster_map' in encoders:
        merchant_map = encoders['merchant_cluster_map']
        if merchant_map:
            values = list(merchant_map.values())
            min_val = min(values)
            max_val = max(values)
            print(f"\n--- Field Specifics: merchant_cluster ---")
            print(f"  Min value: {min_val}")
            print(f"  Max value: {max_val}")
            print(f"  Cardinality: {len(set(values))}")
    
    if 'amt_edges' in encoders:
        amt_edges = encoders['amt_edges']
        if len(amt_edges) > 0:
            print(f"\n--- Field Specifics: amt_bin ---")
            print(f"  Min value: 0")
            print(f"  Max value: {len(amt_edges) - 2}")
            print(f"  Cardinality: {len(amt_edges) - 1}")

    if 'top_cats' in encoders:
        top_cats = encoders['top_cats']
        # +1 for 'other'
        cardinality = len(top_cats) + 1
        print(f"\n--- Field Specifics: category_small_id ---")
        print(f"  Min value: 0")
        print(f"  Max value: {cardinality-1}")
        print(f"  Cardinality: {cardinality}")

    if 'n_geocell_id' in encoders:
        n_geocell_id = encoders['n_geocell_id']
        print(f"\n--- Field Specifics: geocell_id ---")
        print(f"  Min value: 0")
        print(f"  Max value: {n_geocell_id-1}")
        print(f"  Cardinality: {n_geocell_id}")

    if 'time_bucket_hours' in encoders:
        time_bucket_hours = encoders['time_bucket_hours']
        cardinality = 24 // time_bucket_hours
        print(f"\n--- Field Specifics: time_bucket ---")
        print(f"  Min value: 0")
        print(f"  Max value: {cardinality-1}")
        print(f"  Cardinality: {cardinality}")

    assert len(encoders['gender']) > 0
    assert len(encoders['city']) > 0
    assert len(encoders['job']) > 0
    assert len(encoders['age_bucket']) > 0
    print("All y feature encoders are present and not empty.")


if __name__ == '__main__':
    inspect_encoders('data/encoders/encoders.pkl')
