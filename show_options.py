import pickle

print("--- Available Demographic Categories ---")

try:
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    for name, encoder in encoders.items():
        if name.startswith('person_'):
            # Limit the number of classes shown for long lists like city/job
            num_classes_to_show = 10
            class_list = list(encoder.classes_)
            
            print(f"\n# Category: {name}")
            if len(class_list) > num_classes_to_show:
                print(f"  (Showing first {num_classes_to_show} of {len(class_list)} options)")
                for i, class_name in enumerate(class_list[:num_classes_to_show]):
                    print(f"  - {class_name} (ID: {i})")
                print("  ...")
            else:
                for i, class_name in enumerate(class_list):
                    print(f"  - {class_name} (ID: {i})")

except FileNotFoundError:
    print("\nError: `encoders.pkl` not found.")
    print("Please ensure the file is in the root directory.")
