import wfdb

def inspect_atr_file(file_path):
    try:
        # Read annotations from the .atr file
        annotation = wfdb.rdann(file_path, 'atr')
        print("Sample Annotations:", annotation.sample)
        print("Symbols:", annotation.symbol)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# File path to inspect
file_path = "p10/p10981/p10981_s18"
inspect_atr_file(file_path)