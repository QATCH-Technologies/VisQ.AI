from xgb_predictor import XGBPredictor
from nn_predictor import NNPredictor
from linear_predictor import LinearPredictor
import os
import sys
import time
import platform
import psutil
import tempfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# 1) Make sure your project root is importable
sys.path.append(os.getcwd())

# 2) Import all three predictors

# 3) Gather machine specs
specs = {
    'Platform': platform.platform(),
    'Processor': platform.processor(),
    'CPU cores (physical)': psutil.cpu_count(logical=False),
    'CPU max freq (MHz)': psutil.cpu_freq().max,
    'Total RAM (GB)': round(psutil.virtual_memory().total / (1024**3), 2)
}
specs_df = pd.DataFrame([specs])
display(specs_df.style.set_caption("Machine Specifications"))

# 4) Instantiate predictors
model_base = 'visQAI/objects'
predictors = {
    'Linear': LinearPredictor(model_dir=os.path.join(model_base, 'linear_regressor')),
    'NeuralNet': NNPredictor(model_dir=os.path.join(model_base, 'nn_regressor')),
    'XGBoost': XGBPredictor(model_dir=os.path.join(model_base, 'xgb_regressor'))
}

# XGBoost needs a base param dict for incremental updates
xgb_base_params = {
    "objective": "reg:squarederror",
    "seed": 42
}

# 5) Benchmark loop configuration
threshold = 10.0   # seconds
n = 100    # starting batch size
factor = 2      # multiply batch size each iteration

# We'll track when each predictor has crossed the threshold
done = {name: False for name in predictors}
results = []

# 6) Run until every predictor has seen ≥threshold seconds
while not all(done.values()):
    # 6a) Synthesize a new batch of size n
    cat_vals = ['A', 'B', 'C']
    X = pd.DataFrame({
        'Protein type': np.random.choice(cat_vals, n),
        'Protein': np.random.uniform(0, 200, n),
        'Temperature': np.random.uniform(0, 100, n),
        'Buffer': np.random.choice(cat_vals, n),
        'Sugar': np.random.choice(cat_vals, n),
        'Sugar (M)': np.random.uniform(0, 1, n),
        'Surfactant': np.random.choice(cat_vals, n),
        'TWEEN': np.random.uniform(0, 10, n),
    })
    # Grab one of the predictor's target_columns for shape
    sample_pred = next(iter(predictors.values()))
    y = pd.DataFrame(
        np.random.uniform(0, 1000, size=(n, len(sample_pred.target_columns))),
        columns=sample_pred.target_columns
    )
    df_new = pd.concat([X, y], axis=1)

    # 6b) Measure DataFrame in‑memory size
    df_bytes = df_new.memory_usage(deep=True).sum()
    df_mb = df_bytes / (1024**2)

    # 6c) Write to a temp CSV to measure file size
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_name = tmp.name
    df_new.to_csv(tmp_name, index=False)
    file_mb = os.path.getsize(tmp_name) / (1024**2)
    os.remove(tmp_name)

    # 6d) Time each predictor (unless it's already done)
    for name, pred in predictors.items():
        if done[name]:
            continue

        start = time.perf_counter()
        if name == 'XGBoost':
            pred.update(df_new,
                        xgb_params=xgb_base_params,
                        num_boost_round=10,
                        save=False,
                        tune=False)
        else:
            # both Linear and NeuralNet share update(df_new, save=False)
            pred.update(df_new, save=False, tune=False)
        elapsed = time.perf_counter() - start

        # record the result
        results.append({
            'Predictor': name,
            'Batch Size': n,
            'DF Size (MB)': round(df_mb, 2),
            'File Size (MB)': round(file_mb, 2),
            'Update Time (s)': round(elapsed, 3)
        })
        print(
            f"{name:9} | n={n:<6} mem={df_mb:>5.2f}MB file={file_mb:>5.2f}MB time={elapsed:>6.3f}s")

        # mark done if threshold reached
        if elapsed >= threshold:
            done[name] = True

    # increase batch size for next round
    n *= factor

# 7) Show raw results
results_df = pd.DataFrame(results)
display(results_df.style.set_caption("Benchmark Results"))

# 8) Plot: Update Time vs DataFrame Size
fig, ax = plt.subplots()
for name, group in results_df.groupby('Predictor'):
    ax.plot(group['DF Size (MB)'], group['Update Time (s)'],
            marker='o', label=name)
ax.set_xlabel('DataFrame Size (MB)')
ax.set_ylabel('Update Time (s)')
ax.set_title('update() Performance vs. DataFrame Size')
ax.legend()
ax.grid(True)
spec_text = '\n'.join(f"{k}: {v}" for k, v in specs.items())
ax.text(0.05, 0.95, spec_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.3))

# 9) Plot: Update Time vs File Size
fig, ax = plt.subplots()
for name, group in results_df.groupby('Predictor'):
    ax.plot(group['File Size (MB)'],
            group['Update Time (s)'], marker='o', label=name)
ax.set_xlabel('File Size (MB)')
ax.set_ylabel('Update Time (s)')
ax.set_title('update() Performance vs. File Size')
ax.legend()
ax.grid(True)
ax.text(0.05, 0.95, spec_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.3))

plt.show()
