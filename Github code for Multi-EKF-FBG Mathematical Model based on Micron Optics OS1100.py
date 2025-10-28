
"""
===================================================================================
Multi-EKF-FBG Temperature Estimation Model Based on Micron Optics OS1100
===================================================================================

This model simulates a single or multiple Micron Optics OS1100 FBG sensors for temperature 
estimation using Extended Kalman Filter (EKF) and Monte Carlo analysis.

Features:
- Supports both seeded and non-seeded Monte Carlo simulation options
- Configurable number of bonded and reference FBG sensors
- Decouples temperature and strain measurements
- Performs Q-R tuning sweep for EKF process (Q) and measurement (R) noise
- Evaluates ME, MAE, RMSE per FBG and per Q-R combination
- Highlights best Q-R parameters (minimum RMSE)
- Generates plots: error trends, Monte Carlo distributions, sensitivity analysis
- Performs BMS readiness evaluation based on latency and RMSE metrics
- Exports results to CSV and Excel for reproducibility

Sensor characteristics (from Micron Optics OS1100 datasheet):
- Number of FBGs: 1
- FBG Length: 10 mm
- Strain Limit: ±5000 μϵ
- Strain Sensitivity: ±1.2 pm/μϵ
- Operating Temperature Range: -40°C to 120°C (extended to 150°C)
- Thermal Response: ~9.9 pm/°C
- Fiber Type: SMF28-Compatible
- Peak Reflectivity (Rmax): >70%
- FWHM (-3 dB point): 0.25 nm (±0.05 nm)
- Isolation: >15 dB (@ ±0.4 nm around centre wavelength)

Noise parameters:
- Interrogator resolution noise: 1.0 pm (actual)
- Strain noise: 0.5 pm (actual)
- Reference sensor noise: 0.5 pm (actual)

References:
- Kersey, A. D., et al., 1997. "Fiber Grating Sensors," J. Lightwave Technology.
- Measures, R. M., 2001. "Structural Monitoring with Fiber Optic Technology."

Assumptions:
- Gaussian noise applied to Bragg wavelength shifts
- Linear relation between Bragg wavelength and temperature change
- EKF state includes temperature and strain bias
- Monte Carlo simulations assume independent noise per FBG and per temperature
===================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time

# -------------------- CONFIGURATION ------------------------------------
temperature_C_actual = np.arange(25, 151, 5)
n_temps = len(temperature_C_actual)
n_runs = 100 # Monte Carlo runs per Q-R
num_fbgs = 4  # 1 bonded + multiple references

# Sensor characteristics
thermal_response_pm_per_C = 9.9
strain_sensitivity_pm_per_microstrain = 1.2

# Noise parameters
interrogator_noise_pm = 1.0
strain_noise_std_pm = 0.5
reference_noise_std_pm = 0.5

# Applied strain to bonded FBG
applied_strain_microstrain = 100.0

# Q-R sweep values (tuning)
Q_values = [0.001, 0.01, 0.1]
R_values = [0.1, 0.5, 1.0]

# Sensitivity sweep values (for analysis)
sensitivity_values = [8.0, 9.0, 10.0, 11.0, 12.0]  # pm/°C

# -------------------- SEED CONTROL -------------------------------------
seeded = True
seed_value = 32
if seeded:
    np.random.seed(seed_value)

# -------------------- STORAGE ------------------------------------------
results_list = []
errors_records = []

# -------------------- EKF FUNCTION -------------------------------------
def ekf_estimation(bonded, refs, Q_val, R_val, thermal_sens=None):
    if thermal_sens is None:
        thermal_sens = thermal_response_pm_per_C

    n_refs = refs.shape[1]
    x = np.array([25.0, 0.0])
    P = np.eye(2) * 10.0
    Q = np.eye(2) * Q_val
    R = np.eye(1 + n_refs) * R_val

    temp_est = []
    strain_est = []

    for k in range(n_temps):
        x_pred = x.copy()
        P_pred = P + Q

        h_bonded = thermal_sens * (x_pred[0] - 25) + strain_sensitivity_pm_per_microstrain * x_pred[1]
        h_refs = thermal_sens * (x_pred[0] - 25) * np.ones(n_refs)
        h = np.hstack([h_bonded, h_refs])

        H = np.zeros((1 + n_refs, 2))
        H[0, 0] = thermal_sens
        H[0, 1] = strain_sensitivity_pm_per_microstrain
        for j in range(n_refs):
            H[1 + j, 0] = thermal_sens

        z = np.hstack([bonded[k], refs[k, :]])
        y = z - h
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred

        temp_est.append(x[0])
        strain_est.append(x[1])

    return np.array(temp_est), np.array(strain_est)

# -------------------- MONTE CARLO + Q-R SWEEP --------------------------
for Q_val in Q_values:
    for R_val in R_values:
        temp_all_runs = np.zeros((n_runs, n_temps))
        strain_all_runs = np.zeros((n_runs, n_temps))
        for run in range(n_runs):
            if seeded:
                np.random.seed(seed_value + run)

            strain_effect_pm = applied_strain_microstrain * strain_sensitivity_pm_per_microstrain
            bonded_meas = thermal_response_pm_per_C * (temperature_C_actual - 25) + strain_effect_pm
            bonded_meas += np.random.normal(0, strain_noise_std_pm, n_temps)
            bonded_meas += np.random.normal(0, interrogator_noise_pm, n_temps)

            refs_meas = thermal_response_pm_per_C * (temperature_C_actual - 25)[:, np.newaxis]
            refs_meas += np.random.normal(0, reference_noise_std_pm, refs_meas.shape)
            refs_meas += np.random.normal(0, interrogator_noise_pm, refs_meas.shape)

            temp_est, strain_est = ekf_estimation(bonded_meas, refs_meas, Q_val, R_val)
            temp_all_runs[run, :] = temp_est
            strain_all_runs[run, :] = strain_est

        temp_errors = temp_all_runs - temperature_C_actual
        strain_errors = strain_all_runs - applied_strain_microstrain

        ME = np.mean(temp_errors)
        MAE = np.mean(np.abs(temp_errors))
        RMSE = np.sqrt(np.mean(temp_errors**2))

        results_list.append({
            "Q": Q_val, "R": R_val, "ME_mean": ME, "MAE_mean": MAE, "RMSE_mean": RMSE
        })

        errors_df_temp = pd.DataFrame({
            "Q": np.repeat(Q_val, n_runs * n_temps),
            "R": np.repeat(R_val, n_runs * n_temps),
            "Error_C": temp_errors.flatten()
        })
        errors_records.append(errors_df_temp)

results_df = pd.DataFrame(results_list)
errors_df = pd.concat(errors_records, ignore_index=True)

# -------------------- IDENTIFY BEST & TOP-3 -----------------------------
results_sorted = results_df.sort_values("RMSE_mean")
best_row = results_sorted.iloc[0]
best_Q = best_row["Q"]
best_R = best_row["R"]
top3_rows = results_sorted.iloc[:3]

best_summary = pd.DataFrame({
    "Metric": ["Q (process noise)", "R (measurement noise)", "ME (°C)", "MAE (°C)", "RMSE (°C)"],
    "Value": [best_Q, best_R, best_row["ME_mean"], best_row["MAE_mean"], best_row["RMSE_mean"]]
})
print("\nRecommended EKF Tuning Parameters Based on Minimum RMSE")
print(best_summary.to_string(index=False))
print("Top 3 Q-R combinations (by RMSE):")
print(top3_rows[["Q","R","ME_mean","MAE_mean","RMSE_mean"]].to_string(index=False))

# -------------------- PURE FBG vs EKF (Best Q-R) ---------------------
if seeded:
    np.random.seed(seed_value + 999)

strain_effect_pm = applied_strain_microstrain * strain_sensitivity_pm_per_microstrain
bonded_meas = thermal_response_pm_per_C * (temperature_C_actual - 25) + strain_effect_pm
refs_meas = thermal_response_pm_per_C * (temperature_C_actual - 25)[:, np.newaxis]

bonded_meas += np.random.normal(0, strain_noise_std_pm, n_temps)
bonded_meas += np.random.normal(0, interrogator_noise_pm, n_temps)
refs_meas += np.random.normal(0, reference_noise_std_pm, refs_meas.shape)
refs_meas += np.random.normal(0, interrogator_noise_pm, refs_meas.shape)

fbg_est = np.mean(refs_meas, axis=1) / thermal_response_pm_per_C + 25
ekf_temp_est, _ = ekf_estimation(bonded_meas, refs_meas, best_Q, best_R)

def summary_metrics(est, actual):
    error = est - actual
    return np.mean(error), np.mean(np.abs(error)), np.sqrt(np.mean(error**2))

fbg_metrics = summary_metrics(fbg_est, temperature_C_actual)
ekf_metrics = summary_metrics(ekf_temp_est, temperature_C_actual)

summary_df = pd.DataFrame({
    "Method": ["Pure FBG", "EKF (Best Q-R)"],
    "ME (°C)": [fbg_metrics[0], ekf_metrics[0]],
    "MAE (°C)": [fbg_metrics[1], ekf_metrics[1]],
    "RMSE (°C)": [fbg_metrics[2], ekf_metrics[2]]
})
print("\nTemperature Estimation Summary: Pure FBG vs EKF")
print(summary_df.to_string(index=False))

# -------------------- PLOTS DIRECTORY --------------------------
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -------------------- ERROR LINE CHART & BOXPLOT --------------------
best_errors = errors_df[(errors_df["Q"] == best_Q) & (errors_df["R"] == best_R)].copy()
best_errors["Temp_C"] = np.tile(temperature_C_actual, n_runs)
grouped = best_errors.groupby("Temp_C")["Error_C"]
mean_error = grouped.mean()
std_error = grouped.std()

plt.figure(figsize=(16, 12))

# Subplot 1: Temperature error
plt.figure(figsize=(16, 12))
plt.plot(temperature_C_actual, fbg_est - temperature_C_actual, label="Pure FBG", color="blue", marker='o')
plt.plot(temperature_C_actual, ekf_temp_est - temperature_C_actual, label="EKF (Best Q-R)", color="red", marker='s')
plt.fill_between(temperature_C_actual, mean_error - 2*std_error, mean_error + 2*std_error, color="red", alpha=0.2)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Temperature Error (°C)")
plt.title("Temperature Estimation Error: Pure FBG vs EKF")
plt.grid(True)
plt.legend()

# Subplot 2: Monte Carlo error distribution
plt.figure(figsize=(16, 12))
rmse_matrix = best_errors["Error_C"].values.reshape(n_runs, n_temps)
rmse_df = pd.DataFrame(rmse_matrix, columns=temperature_C_actual)
sns.boxplot(data=rmse_df, whis=[5,95])
sns.swarmplot(data=rmse_df, color=".25", size=3)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Temperature Error (°C)")
plt.title("Monte Carlo Temperature Error Distribution (EKF)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "MC_error_distribution.png"))
plt.show()

# -------------------- SENSITIVITY ANALYSIS -----------------
sensitivity_results = []
for sens in sensitivity_values:
    if seeded:
        np.random.seed(seed_value + 999)
    strain_effect_pm = applied_strain_microstrain * strain_sensitivity_pm_per_microstrain
    bonded_meas = sens * (temperature_C_actual - 25) + strain_effect_pm
    refs_meas = sens * (temperature_C_actual - 25)[:, np.newaxis]
    bonded_meas += np.random.normal(0, strain_noise_std_pm, n_temps)
    bonded_meas += np.random.normal(0, interrogator_noise_pm, n_temps)
    refs_meas += np.random.normal(0, reference_noise_std_pm, refs_meas.shape)
    refs_meas += np.random.normal(0, interrogator_noise_pm, refs_meas.shape)

    ekf_temp_est_sens, _ = ekf_estimation(bonded_meas, refs_meas, best_Q, best_R, thermal_sens=sens)
    temp_error = ekf_temp_est_sens - temperature_C_actual
    RMSE = np.sqrt(np.mean(temp_error**2))
    sensitivity_results.append({"Thermal_Sensitivity": sens, "RMSE_C": RMSE})

sensitivity_df = pd.DataFrame(sensitivity_results)
plt.figure(figsize=(16, 12))
plt.plot(sensitivity_df["Thermal_Sensitivity"], sensitivity_df["RMSE_C"], marker='o', color='green')
plt.xlabel("FBG Thermal Sensitivity (pm/°C)")
plt.ylabel("RMSE (°C)")
plt.title("Sensitivity of EKF Temperature Estimation to FBG Thermal Response")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "sensitivity.png"))
plt.show()

errors_df["Temp_C"] = np.tile(temperature_C_actual, n_runs * len(Q_values) * len(R_values))

# -------------------- Q-R ERROR SWEEP LINE PLOT --------------------
plt.figure(figsize=(12, 8))

qr_rmse = errors_df.groupby(["Q", "R", "Temp_C"])["Error_C"].apply(
    lambda x: np.sqrt(np.mean(x**2))
).reset_index()

# Line plot for each Q-R combo
for (q_val, r_val), subset in qr_rmse.groupby(["Q", "R"]):
    label = f"Q={q_val}, R={r_val}"
    plt.plot(subset["Temp_C"], subset["Error_C"], marker='o', label=label)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("RMSE (°C)")
plt.title("Q-R Error Sweep vs Actual Temperature")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "qr_error_sweep_line.png"))
plt.show()

#error plot

plt.figure(figsize=(18, 14))

# Top-left: Temperature error
plt.subplot(2, 2, 1)
plt.plot(temperature_C_actual, fbg_est - temperature_C_actual, label="Pure FBG", color="blue", marker='o')
plt.plot(temperature_C_actual, ekf_temp_est - temperature_C_actual, label="EKF (Best Q-R)", color="red", marker='s')
plt.fill_between(temperature_C_actual, mean_error - 2*std_error, mean_error + 2*std_error, color="red", alpha=0.2)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Temperature Error (°C)")
plt.title("Temperature Estimation Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "Temperature_estimation_error.png"))
plt.show()

# -------------------- ADD TO COMBINED PLOT --------------------
plt.figure(figsize=(18, 14))

# Top-left: Temperature error
plt.subplot(2, 2, 1)
plt.plot(temperature_C_actual, fbg_est - temperature_C_actual, label="Pure FBG", color="blue", marker='o')
plt.plot(temperature_C_actual, ekf_temp_est - temperature_C_actual, label="EKF (Best Q-R)", color="red", marker='s')
plt.fill_between(temperature_C_actual, mean_error - 2*std_error, mean_error + 2*std_error, color="red", alpha=0.2)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Temperature Error (°C)")
plt.title("Temperature Estimation Error")
plt.grid(True)
plt.legend()


# Top-right: Monte Carlo error distribution
plt.subplot(2, 2, 2)
sns.boxplot(data=rmse_df, whis=[5,95])
sns.swarmplot(data=rmse_df, color=".25", size=3)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Temperature Error (°C)")
plt.title("Monte Carlo Temperature Error Distribution (EKF)")
plt.xticks(rotation=45)

# Bottom-left: Sensitivity analysis
plt.subplot(2, 2, 3)
plt.plot(sensitivity_df["Thermal_Sensitivity"], sensitivity_df["RMSE_C"], marker='o', color='green')
plt.xlabel("FBG Thermal Sensitivity (pm/°C)")
plt.ylabel("RMSE (°C)")
plt.title("Sensitivity of EKF Estimation to FBG Thermal Response")
plt.grid(True)

# Bottom-right: Q-R sweep line plot
plt.subplot(2, 2, 4)
for (q_val, r_val), subset in qr_rmse.groupby(["Q", "R"]):
    label = f"Q={q_val}, R={r_val}"
    plt.plot(subset["Temp_C"], subset["Error_C"], marker='o', label=label)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("RMSE (°C)")
plt.title("Q-R Error Sweep vs Actual Temperature")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "combined_with_qr_sweep_line.png"))
plt.show()

# -------------------- MEASURE EKF LATENCY --------------------------
# Use a single run with best Q-R parameters (re-use bonded_meas & refs_meas above as representative)
if seeded:
    np.random.seed(seed_value + 999)

# Generate sample bonded and reference measurements for timing (use thermal_response_pm_per_C)
strain_effect_pm = applied_strain_microstrain * strain_sensitivity_pm_per_microstrain
bonded_meas_t = thermal_response_pm_per_C * (temperature_C_actual - 25) + strain_effect_pm
refs_meas_t = thermal_response_pm_per_C * (temperature_C_actual - 25)[:, np.newaxis]
bonded_meas_t += np.random.normal(0, strain_noise_std_pm, n_temps)
bonded_meas_t += np.random.normal(0, interrogator_noise_pm, n_temps)
refs_meas_t += np.random.normal(0, reference_noise_std_pm, refs_meas_t.shape)
refs_meas_t += np.random.normal(0, interrogator_noise_pm, refs_meas_t.shape)

# Time EKF execution (measure whole run)
start_time = time.time()
ekf_temp_est_t, _ = ekf_estimation(bonded_meas_t, refs_meas_t, best_Q, best_R)
end_time = time.time()

latency_ms = (end_time - start_time) * 1000  # convert seconds to milliseconds
print(f"\nEstimated EKF latency (whole sweep): {latency_ms:.2f} ms")

# -------------------- DETAILED RESPONSE TIME PER STEP --------------------------
step_latencies = []

# Measure per-step EKF timing
x = np.array([25.0, 0.0])
P = np.eye(2) * 10.0
Q = np.eye(2) * best_Q
R = np.eye(1 + refs_meas_t.shape[1]) * best_R

for k in range(n_temps):
    start_step = time.time()

    # Single step EKF calculation
    x_pred = x.copy()
    P_pred = P + Q

    h_bonded = thermal_response_pm_per_C * (x_pred[0] - 25) + strain_sensitivity_pm_per_microstrain * x_pred[1]
    h_refs = thermal_response_pm_per_C * (x_pred[0] - 25) * np.ones(refs_meas_t.shape[1])
    h = np.hstack([h_bonded, h_refs])

    H = np.zeros((1 + refs_meas_t.shape[1], 2))
    H[0, 0] = thermal_response_pm_per_C
    H[0, 1] = strain_sensitivity_pm_per_microstrain
    for j in range(refs_meas_t.shape[1]):
        H[1 + j, 0] = thermal_response_pm_per_C

    z = np.hstack([bonded_meas_t[k], refs_meas_t[k, :]])
    y = z - h
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    end_step = time.time()
    step_latencies.append((end_step - start_step) * 1000)  # ms

step_latencies = np.array(step_latencies)
total_run_latency = np.sum(step_latencies)
mean_latency = np.mean(step_latencies)
max_latency = np.max(step_latencies)
min_latency = np.min(step_latencies)

# -------------------- SHOW RESULTS TO CONSOLE --------------------------
print("\nEKF Response Time Analysis (per-step and summary):")
for idx, t in enumerate(step_latencies):
    print(f" Step {idx+1:03d}: {t:.4f} ms")
print(f"Total run latency (all steps): {total_run_latency:.4f} ms")
print(f"Mean step latency: {mean_latency:.4f} ms")
print(f"Max step latency: {max_latency:.4f} ms")
print(f"Min step latency: {min_latency:.4f} ms")

# -------------------- SAVE TO CSV --------------------------
response_time_df = pd.DataFrame({
    "Temperature_C": temperature_C_actual,
    "Step_Latency_ms": step_latencies
})
response_time_df["Total_Run_ms"] = total_run_latency
response_time_df["Mean_Step_ms"] = mean_latency
response_time_df["Max_Step_ms"] = max_latency
response_time_df["Min_Step_ms"] = min_latency

os.makedirs("results", exist_ok=True)
response_time_csv_path = os.path.join("results", "ekf_response_times.csv")
response_time_df.to_csv(response_time_csv_path, index=False, sep=',')
print(f"\nPer-step EKF response times saved to CSV: {response_time_csv_path}")

# -------------------- PLOT RESPONSE TIME --------------------------
plt.figure(figsize=(10,5))
plt.plot(temperature_C_actual, step_latencies, marker='o', color='tab:blue', label='Per-Step Latency (ms)')
plt.hlines([mean_latency], xmin=temperature_C_actual[0], xmax=temperature_C_actual[-1],
           colors='green', linestyles='--', label=f'Mean Latency: {mean_latency:.2f} ms')
plt.hlines([max_latency], xmin=temperature_C_actual[0], xmax=temperature_C_actual[-1],
           colors='red', linestyles='--', label=f'Max Latency: {max_latency:.2f} ms')
plt.hlines([min_latency], xmin=temperature_C_actual[0], xmax=temperature_C_actual[-1],
           colors='orange', linestyles='--', label=f'Min Latency: {min_latency:.2f} ms')
plt.xlabel("Temperature (°C)")
plt.ylabel("EKF Step Response Time (ms)")
plt.title("Detailed EKF Per-Step Response Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("plots", "ekf_response_time_plot.png"))
plt.show()


# -------------------- BMS READINESS CHECK (single, data-driven) --------------------------
# Build report using measured latency
bms_report_measured = {
    "Latency_ms": latency_ms,
    "Noise_Robust_RMSE_mean": ekf_metrics[2],
    "Multi_FBG_RMSE": ekf_metrics[2],
    "Sensitivity": sensitivity_results
}

def plot_bms_readiness_and_scores(report, plots_dir="plots"):
    latency_ms = report['Latency_ms']
    noise_rmse = report['Noise_Robust_RMSE_mean']
    multi_fbg_rmse = report['Multi_FBG_RMSE']
    sensitivity_df_local = pd.DataFrame(report['Sensitivity'])

    # Original (direct) scheme: normalized metrics then overall = 1 - mean(norms)
    latency_norm = min(1.0, latency_ms / 50)               # higher is worse
    noise_norm = min(1.0, noise_rmse / 0.2)               # higher is worse
    multi_fbg_norm = min(1.0, multi_fbg_rmse / 0.2)       # higher is worse
    sensitivity_norm = min(1.0, sensitivity_df_local['RMSE_C'].max() / 0.2)  # higher is worse

    direct_overall_score = 1 - np.mean([latency_norm, noise_norm, multi_fbg_norm, sensitivity_norm])
    # Inverted (intuitive higher-is-better) per-submetric then averaged
    latency_score_inv = 1 - latency_norm
    noise_score_inv = 1 - noise_norm
    multi_fbg_score_inv = 1 - multi_fbg_norm
    sensitivity_score_inv = 1 - sensitivity_norm
    inverse_overall_score = np.mean([latency_score_inv, noise_score_inv, multi_fbg_score_inv, sensitivity_score_inv])

    # Console prints
    print("\nBMS Readiness (data-driven):")
    print(f" Latency (ms): {latency_ms:.2f}")
    print(f" Noise RMSE (°C): {noise_rmse:.6f}")
    print(f" Multi-FBG RMSE (°C): {multi_fbg_rmse:.6f}")
    print(f" Max Sensitivity RMSE (°C): {sensitivity_df_local['RMSE_C'].max():.6f}")
    print(f"\n Direct overall score (1=best): {direct_overall_score:.4f}")
    print(f" Inverse overall score (1=best): {inverse_overall_score:.4f}")

    # Bar chart of raw metrics (values)
    plt.figure(figsize=(10,6))
    metrics = ['Latency (ms)', 'Noise RMSE (°C)', 'Multi-FBG RMSE (°C)', 'Max Sensitivity RMSE (°C)']
    values = [latency_ms, noise_rmse, multi_fbg_rmse, sensitivity_df_local['RMSE_C'].max()]
    plt.bar(metrics, values, color=['orange','red','blue','green'])
    plt.title(f"BMS Readiness Metrics (Direct score: {direct_overall_score:.2f}, Inverse score: {inverse_overall_score:.2f})")
    plt.ylabel("Metric Value")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01*max(values), f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bms_readiness_metrics.png"))
    plt.show()

    # Sensitivity plot for BMS
    plt.figure(figsize=(8,5))
    plt.plot(sensitivity_df_local['Thermal_Sensitivity'], sensitivity_df_local['RMSE_C'], marker='o', color='purple')
    plt.title("EKF Temperature Sensitivity Analysis for BMS")
    plt.xlabel("FBG Thermal Sensitivity (pm/°C)")
    plt.ylabel("RMSE (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bms_sensitivity_plot.png"))
    plt.show()

    # Latency vs RMSE scatter
    plt.figure(figsize=(6,4))
    plt.scatter(latency_ms, noise_rmse, color='tab:red', s=80)
    plt.xlabel("Latency (ms)")
    plt.ylabel("EKF RMSE (°C)")
    plt.title("Latency vs EKF Accuracy (RMSE)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "latency_vs_rmse.png"))
    plt.show()

    # Return both scores and key values for export
    return {
        "direct_overall_score": direct_overall_score,
        "inverse_overall_score": inverse_overall_score,
        "latency_ms": latency_ms,
        "noise_rmse": noise_rmse,
        "multi_fbg_rmse": multi_fbg_rmse,
        "max_sensitivity_rmse": sensitivity_df_local['RMSE_C'].max()
    }

# Run single, consolidated BMS readiness + plots + scores
bms_scores_dict = plot_bms_readiness_and_scores(bms_report_measured)

# -------------------- EXPORT ALL RESULTS TO SINGLE EXCEL FILE --------------------------
excel_path = os.path.join("results", "ekf_fbg_analysis.xlsx")
os.makedirs("results", exist_ok=True)

with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    # 1. Monte Carlo Q-R sweep results
    results_df.to_excel(writer, sheet_name="QR_Sweep_Results", index=False)

    # 2. Error records for each run
    errors_df.to_excel(writer, sheet_name="Errors_All_Runs", index=False)

    # 3. Pure FBG vs EKF summary metrics
    summary_df.to_excel(writer, sheet_name="FBG_vs_EKF", index=False)

    # 4. Sensitivity analysis
    sensitivity_df.to_excel(writer, sheet_name="Sensitivity_Analysis", index=False)

    # 5. BMS readiness metrics & scores
    bms_export_df = pd.DataFrame({
        "Metric": [
            'Latency (ms)',
            'Noise RMSE (°C)',
            'Multi-FBG RMSE (°C)',
            'Max Sensitivity RMSE (°C)',
            'Direct Overall Score (1=best)',
            'Inverse Overall Score (1=best)'
        ],
        "Value": [
            bms_scores_dict["latency_ms"],
            bms_scores_dict["noise_rmse"],
            bms_scores_dict["multi_fbg_rmse"],
            bms_scores_dict["max_sensitivity_rmse"],
            bms_scores_dict["direct_overall_score"],
            bms_scores_dict["inverse_overall_score"]
        ]
    })
    bms_export_df.to_excel(writer, sheet_name="BMS_Readiness", index=False)

    # 6. Add latency vs RMSE data (small table) for traceability
    latency_vs_rmse_df = pd.DataFrame({
        "Latency_ms": [bms_scores_dict["latency_ms"]],
        "EKF_RMSE_C": [bms_scores_dict["noise_rmse"]]
    })
    latency_vs_rmse_df.to_excel(writer, sheet_name="Latency_vs_RMSE", index=False)

print(f"\nAll EKF-FBG results exported to Excel file: {excel_path}")

# =============================================================================
# END OF SCRIPT SUMMARY
# =============================================================================
"""
This script performs temperature estimation for single or multiple FBG sensors using
an EKF-based approach. It supports Monte Carlo simulation for robust error statistics,
Q-R tuning analysis, and sensitivity sweeps. It generates multiple plots and exports
all relevant data to CSV/Excel for reproducibility. The script is organized into
three main sections:

1. EKF Estimation & Monte Carlo: Handles all computations, Q-R sweeps, and error metrics.
2. Plotting: All figures for error trends, Monte Carlo distributions, and sensitivity analysis.
3. Logging & Saving: All results exported to files; console outputs are grouped at the top.

The code is structured to maintain clarity, modularity, and reproducibility
# =============================================================================
"""
# End of script.
