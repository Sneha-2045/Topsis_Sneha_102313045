import sys
import pandas as pd
import numpy as np

# -----------------------------
# Validation Functions
# -----------------------------
def validate_inputs(args):
    if len(args) != 5:
        print("❌ Usage:")
        print("python topsis.py <InputDataFile> <Weights> <Impacts> <OutputFile>")
        sys.exit(1)

def validate_file(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print("❌ File not found:", filename)
        sys.exit(1)

def validate_numeric(df):
    if df.shape[1] < 3:
        print("❌ Input file must contain at least 3 columns.")
        sys.exit(1)
    
    try:
        df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    except:
        print("❌ From 2nd column onward all values must be numeric.")
        sys.exit(1)

def validate_weights_impacts(weights, impacts, n_cols):
    if len(weights) != n_cols or len(impacts) != n_cols:
        print("❌ Number of weights and impacts must match number of criteria.")
        sys.exit(1)

    for i in impacts:
        if i not in ['+','-']:
            print("❌ Impacts must be either + or -.")
            sys.exit(1)

# -----------------------------
# TOPSIS Logic
# -----------------------------
def topsis(df, weights, impacts):
    data = df.iloc[:,1:].values.astype(float)

    # Step 1: Normalize
    norm = np.sqrt((data ** 2).sum(axis=0))
    normalized = data / norm

    # Step 2: Weighted normalized matrix
    weighted = normalized * weights

    # Step 3: Ideal best & worst
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted[:,i].max())
            ideal_worst.append(weighted[:,i].min())
        else:
            ideal_best.append(weighted[:,i].min())
            ideal_worst.append(weighted[:,i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Distance
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Score
    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False)

    return df

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    
    validate_inputs(sys.argv)

    input_file = sys.argv[1]
    weights = list(map(float, sys.argv[2].split(",")))
    impacts = sys.argv[3].split(",")
    output_file = sys.argv[4]

    df = validate_file(input_file)
    validate_numeric(df)

    n_cols = df.shape[1] - 1
    validate_weights_impacts(weights, impacts, n_cols)

    result = topsis(df, weights, impacts)
    result.to_csv(output_file, index=False)

    print("✅ TOPSIS completed successfully!")
    print("📁 Output saved to:", output_file)
