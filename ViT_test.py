import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from ViT_training import ViTMultiHead  # Updated import

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Antenna Parameters (must match dataset.py) ===
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5

# === Load Test Data ===
test_images = np.load(os.path.join(script_dir, 'Minput_images_polar_test.npy'))  # (~525, 224, 224, 3)
test_targets = np.load(os.path.join(script_dir, 'Moutput_test.npy')).T  # (15, 5)
test_features = np.load(os.path.join(script_dir, 'Minput_test.npy')).T  # (15, 4)

# === Load Model ===
model = ViTMultiHead(num_rings=5, num_classes_per_ring=11, pretrained=False, dropout_rate=0.5).to(DEVICE)  # Added dropout_rate

model_path = os.path.join(script_dir, 'best_vit_antenna_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print(f"Model not found at {model_path}. Using random initialization.")

model.eval()

# === Antenna Pattern Calculation ===
def calculate_radiation_pattern(elements_per_ring, theta0deg):
    
    theta0 = np.deg2rad(theta0deg)
    phi0 = 0
    phi = 0
    theta = np.linspace(0, 2 * np.pi, 1000)
    theta_deg = np.rad2deg(theta)
    
    radii = r0 + delta_r * np.arange(max_rings)
    AF_az = np.zeros_like(theta, dtype=complex)
    
    for ring in range(max_rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)
    
    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    
    # Main lobe gain
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    main_lobe_gain_dB = 20 * np.log10(maxVal + np.finfo(float).eps)
    
    # HPBW
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)
    
    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(AF_dB_ext[:maxIdx_ext] <= halfPower) else None
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
    
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]
    
    # Side lobe level
    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)
    pk = responseLin[peaks]
    
    if len(pk) == 0:
        SSL_gain_dB = 0
    else:
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        threshold_dB = 1
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
        SSL_gain_dB = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(side_lobe_idx) > 0 else 0
    
    return AF_dB_az, theta_deg, HPBW, main_lobe_gain_dB, SSL_gain_dB


# === Inference & Evaluation ===

all_accuracies = []
per_ring_accuracies = [[] for _ in range(5)]
predictions_list = []
targets_list = []

with torch.no_grad():
    for idx in range(len(test_images)):
        # Load image
        img = test_images[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)  # (1, 3, 240, 240)
        
        # Inference
        outputs = model(img)  # list of (1, 11)
        
        # Get predictions
        predictions = [out.argmax(dim=1).item() for out in outputs]
        targets = test_targets[idx].astype(int)
        
        predictions_list.append(predictions)
        targets_list.append(targets)
        
        # Accuracy per ring
        for ring in range(5):
            match = 1 if predictions[ring] == targets[ring] else 0
            per_ring_accuracies[ring].append(match)
        
        # Overall accuracy (all 5 rings must match)
        overall_match = 1 if all(predictions[r] == targets[r] for r in range(5)) else 0
        all_accuracies.append(overall_match)

# Compute metrics
overall_acc = np.mean(all_accuracies) * 100
per_ring_acc = [np.mean(accs) * 100 for accs in per_ring_accuracies]

print("Evaluation Results")
print(f"Overall Accuracy (all rings correct): {overall_acc:.2f}%")
print("Per-ring Accuracy:")
for ring in range(5):
    print(f"  Ring {ring+1}: {per_ring_acc[ring]:.2f}%")

# === Detailed Comparison for First Sample ===
print("Detailed Results for Test Sample 0")

sample_idx = 0
theta0deg = test_features[sample_idx, 3]
pred_arch = np.array(predictions_list[sample_idx])
true_arch = test_targets[sample_idx].astype(int)

print(f"\nTheta0: {theta0deg:.2f}°")
print(f"\nArchitecture Prediction vs Ground Truth:")
print(f"  Ring | Predicted | True")
for ring in range(5):
    match = "✓" if pred_arch[ring] == true_arch[ring] else "✗"
    print(f"   {ring+1}  |    {pred_arch[ring]:2d}     |  {true_arch[ring]:2d}  {match}")

# Compute radiation patterns
AF_pred_dB, theta_deg, HPBW_pred, gain_main_pred, SSL_pred = calculate_radiation_pattern(pred_arch, theta0deg)
AF_true_dB, theta_deg, HPBW_true, gain_main_true, SSL_true = calculate_radiation_pattern(true_arch, theta0deg)

print(f"\nRadiation Pattern Metrics:")
print(f"{'Metric':<25} {'Predicted':<15} {'Ground Truth':<15} {'Error':<10}")
print(f"{'-'*65}")
print(f"{'Main Lobe Gain (dB)':<25} {gain_main_pred:>6.2f} dB         {gain_main_true:>6.2f} dB         {abs(gain_main_pred - gain_main_true):>6.2f} dB")
print(f"{'Side Lobe Level (dB)':<25} {SSL_pred:>6.2f} dB         {SSL_true:>6.2f} dB         {abs(SSL_pred - SSL_true):>6.2f} dB")
print(f"{'HPBW (degrees)':<25} {HPBW_pred:>6.2f}°          {HPBW_true:>6.2f}°          {abs(HPBW_pred - HPBW_true):>6.2f}°")

# === Plot Comparison ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cartesian plot
axes[0].plot(theta_deg, AF_true_dB, label='Ground Truth', linewidth=2, linestyle='-')
axes[0].plot(theta_deg, AF_pred_dB, label='Prediction', linewidth=2, linestyle='--')
axes[0].set_xlabel('Azimuth (°)')
axes[0].set_ylabel('Gain (dB)')
axes[0].set_title('Radiation Pattern - Cartesian')
axes[0].set_xlim([0, 360])
axes[0].set_ylim([-40, 0])
axes[0].grid(True)
axes[0].legend()

# Polar plot
theta_rad = np.deg2rad(theta_deg)
ax_polar = fig.add_subplot(122, projection='polar')
ax_polar.plot(theta_rad, AF_true_dB, label='Ground Truth', linewidth=2)
ax_polar.plot(theta_rad, AF_pred_dB, label='Prediction', linewidth=2, linestyle='--')
ax_polar.set_ylim([-40, 0])
ax_polar.set_title('Radiation Pattern - Polar')
ax_polar.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'vit_test_comparison.png'))
print(f"\n✓ Comparison plot saved to vit_test_comparison.png")
plt.show()

print("\n✅ Evaluation complete!")
