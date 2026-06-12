import pandas as pd
import numpy as np

# 1. Load the data
# Replace with your actual CSV file path
csv_file_path = 'bench1a_sem_scaling.csv'
df = pd.read_csv(csv_file_path)

# Sort by N to ensure logical consistency
df = df.sort_values(by='N').reset_index(drop=True)

print("="*60)
print("          STATISTICAL METRICS VALIDATION REPORT          ")
print("="*60)

# -------------------------------------------------------------
# Metric 1: Slope of log(SEM) vs log(N) == -0.50 ± 0.02
# -------------------------------------------------------------
log_N = np.log10(df['N'])
log_SEM = np.log10(df['sem'])

# Perform linear regression to get the slope (b) and covariance matrix
slope, intercept = np.polyfit(log_N, log_SEM, 1)

# Optional: Calculate standard error of the slope to ensure fit quality
n_points = len(df)
if n_points > 2:
    residuals = log_SEM - (slope * log_N + intercept)
    residual_sum_of_squares = np.sum(residuals**2)
    degrees_of_freedom = n_points - 2
    x_mean = np.mean(log_N)
    slope_err = np.sqrt((residual_sum_of_squares / degrees_of_freedom) / np.sum((log_N - x_mean)**2))
else:
    slope_err = 0.0

slope_pass = -0.52 <= slope <= -0.48
status_1 = "PASS" if slope_pass else "FAIL"

print(f"1. Slope of log(SEM) vs log(N):")
print(f"   - Target: -0.50 ± 0.02")
print(f"   - Measured: {slope:.4f} (Fit Standard Error: ±{slope_err:.4f})")
print(f"   - Status: [{status_1}]")
print("-" * 60)

# -------------------------------------------------------------
# Metric 2: sem * sqrt(N) constancy within ~10%
# -------------------------------------------------------------
# Calculate the metric using the 'sem_sqrtN' column or deriving it directly
sem_sqrtN = df['sem'] * np.sqrt(df['N'])

# We check constancy by seeing how much the values deviate from their mean or median
# Variation index: (Max - Min) / Mean  OR  Relative Standard Deviation (RSD)
mean_val = np.mean(sem_sqrtN)
max_deviation = (np.max(sem_sqrtN) - np.min(sem_sqrtN)) / mean_val * 100
rsd = (np.std(sem_sqrtN, ddof=1) / mean_val) * 100

# Using Max-Min span relative to mean as a strict 10% bounds check
constancy_pass = max_deviation <= 10.0
status_2 = "PASS" if constancy_pass else "WARNING / FAIL (Check fluctuation)"

print(f"2. sem * sqrt(N) Constancy:")
print(f"   - Target: Fluctuation within ~10%")
print(f"   - Measured Max-to-Min Spread: {max_deviation:.2f}%")
print(f"   - Relative Standard Deviation (RSD): {rsd:.2f}%")
print(f"   - Status: [{status_2}]")
print("-" * 60)

# -------------------------------------------------------------
# Metric 3: Points inside 3σ (SEM) envelope > 90%
# -------------------------------------------------------------
# Count how many absolute errors are strictly less than or equal to 3 * SEM
inside_envelope = df['abs_error'] <= (3 * df['sem'])
percentage_inside = (inside_envelope.sum() / len(df)) * 100

envelope_pass = percentage_inside >= 90.0
status_3 = "PASS" if envelope_pass else "FAIL"

print(f"3. Confidence Envelope Coverage:")
print(f"   - Target: >90% of points inside 3 * SEM envelope")
print(f"   - Measured: {percentage_inside:.2f}% ({inside_envelope.sum()}/{len(df)} points)")
print(f"   - Status: [{status_3}]")
print("-" * 60)

# -------------------------------------------------------------
# Metric 4: Mean convergence approaches exact value
# -------------------------------------------------------------
# Check if the absolute error decreases as N increases
# We can evaluate this by checking if the error at the largest N is smaller than at the smallest N,
# or checking for a negative correlation between N and abs_error.
first_errors = df['abs_error'].head(max(1, len(df)//3)).mean()
last_errors = df['abs_error'].tail(max(1, len(df)//3)).mean()

correlation = df['N'].corr(df['abs_error'], method='spearman')

convergence_pass = last_errors < first_errors and correlation < 0
status_4 = "PASS" if convergence_pass else "MONITOR (Requires larger N or more trials)"

print(f"4. Mean Convergence Profile:")
print(f"   - Target: Error approaches 0 as N increases")
print(f"   - Avg Error (Small N): {first_errors:.6f} vs Avg Error (Large N): {last_errors:.6f}")
print(f"   - Spearman Rank Correlation (N vs Error): {correlation:.4f} (Negative is expected)")
print(f"   - Status: [{status_4}]")
print("="*60)
