import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import scipy.io
import os
import platform
import sys
from PIL import Image



script_dir = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use("Agg")   # prevents plots from popping up



nb_samples = 10000
max_rings = 5
max_elements = 10
theta0_max_deg = 180
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
np.random.seed(46) 

Moutput = np.zeros((max_rings, nb_samples))
Minput = np.zeros((4, nb_samples))
image_h = 224
image_w = 224
images_dir = os.path.join(script_dir, "images")
os.makedirs(images_dir, exist_ok=True)
Minput_images = np.zeros((nb_samples, image_h, image_w, 3), dtype=np.uint8)
Minput_images_polar = np.zeros((nb_samples, image_h, image_w, 3), dtype=np.uint8)

for sample_idx in range(nb_samples):
    
    # --- Architecture complètement aléatoire ---
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:  # S'assurer qu'au moins un ring est actif
            break
    Moutput[:, sample_idx] = elements_per_ring

    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)
    phi0 = 0

    radii = r0 + delta_r * np.arange(max_rings)

    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = 0
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
    
    theta_deg = np.rad2deg(theta)

    try:
        r = AF_norm_az

        fig_polar = plt.figure(figsize=(3, 3), dpi=80, facecolor='white')
        ax = fig_polar.add_subplot(111, projection='polar', facecolor='white')
        ax.plot(theta, r, linewidth=2, color='C0')
        ax.set_ylim([0, 1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)
        try:
            ax.spines['polar'].set_visible(False)
        except Exception:
            ax.set_frame_on(False)
        fig_polar.canvas.draw()
        buf_p = fig_polar.canvas.buffer_rgba()
        img_rgba_p = np.asarray(buf_p, dtype=np.uint8)
        img_rgba_p = img_rgba_p.reshape(fig_polar.canvas.get_width_height()[::-1] + (4,))
        img_polar = img_rgba_p[:, :, :3]
        img_polar = Image.fromarray(img_polar).resize((image_w, image_h))
        img_polar = np.array(img_polar)
        plt.close(fig_polar)

        Minput_images_polar[sample_idx] = img_polar

        img_pil_p = Image.fromarray(img_polar)
        img_filename_p = os.path.join(images_dir, f"sample_{sample_idx:03d}_polar.png")
        img_pil_p.save(img_filename_p)
    except Exception as e:
        pass
    #calcul du gain max non normalisé du lobe principale
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
    
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)
    
    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None
    
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    
        
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
    
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)
    
    pk = responseLin[peaks]
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0
    else: 
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        sorted_angles = theta_deg[peaks][sorted_idx]

        threshold_dB = 1
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]#
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)#
        
        main_lobe_gain = maxVal_non_norm
        
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
        
    
    # --- Stockage ---
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg  



# === Enregistrement de Minput et Moutput en format .mat dans le dossier contenant le code===
script_dir = os.path.dirname(os.path.abspath(__file__))
np.save(os.path.join(script_dir, "Minput_images.npy"), Minput_images)
# Save polar image array as well
np.save(os.path.join(script_dir, "Minput_images_polar.npy"), Minput_images_polar)

# Chemin du fichier .mat à enregistrer
fichier_path_Minput_mat = os.path.join(script_dir, 'Minput.mat')

# Enregistrement de la matrice Minput (anciennement Moutput) au format .mat
scipy.io.savemat(fichier_path_Minput_mat, {'Minput': Minput})

# Chemin du fichier .mat à enregistrer
fichier_path_Moutput_mat = os.path.join(script_dir, 'Moutput.mat')

# Enregistrement de la matrice Moutput au format .mat
scipy.io.savemat(fichier_path_Moutput_mat, {'Moutput': Moutput})

    
    
    
# === Enregistrement de Minput et Moutput en format .npy dans le dossier contenant le code===
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, 'Minput.npy')
np.save(file_path, Minput)

file_path = os.path.join(script_dir, 'Moutput.npy')
np.save(file_path, Moutput)



print("✅ Minput.npy et Moutput.npy ont été enregistrés dans le dossier courant.")