import numpy as np
import scipy.io
from scipy.io import loadmat, savemat
import os
import platform
import sys


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console(): 
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()    
    
np.random.seed(46) 

# === Chargement des matrices Minput et Moutput ===
script_dir = os.path.dirname(os.path.abspath(__file__))

file_path_input = os.path.join(script_dir, 'Minput.npy')
Minput = np.load(file_path_input)

file_path_output = os.path.join(script_dir, 'Moutput.npy')
Moutput = np.load(file_path_output)

# === Chargement des images (si disponibles) ===
try:
    file_path_images = os.path.join(script_dir, 'Minput_images.npy')
    Minput_images = np.load(file_path_images)
    print(f"✓ Minput_images.npy loaded: shape {Minput_images.shape}")
except FileNotFoundError:
    Minput_images = None
    print("⚠ Minput_images.npy not found — skipping image split")

try:
    file_path_images_polar = os.path.join(script_dir, 'Minput_images_polar.npy')
    Minput_images_polar = np.load(file_path_images_polar)
    print(f"✓ Minput_images_polar.npy loaded: shape {Minput_images_polar.shape}")
except FileNotFoundError:
    Minput_images_polar = None
    print("⚠ Minput_images_polar.npy not found — skipping polar image split")

# === Mélange aléatoire des indices ===
total_samples = Minput.shape[1]
rand_indices = np.random.permutation(total_samples)

# === Séparation 85% training / 15% test ===
nb_test = round(0.15 * total_samples)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

print(f"\nTotal samples: {total_samples}")
print(f"Training samples: {len(train_indices)}")
print(f"Test samples: {len(test_indices)}")

# === Sélection des colonnes pour Minput et Moutput ===
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]

# === Sélection des images (si disponibles) ===
if Minput_images is not None:
    Minput_images_test = Minput_images[test_indices]
    Minput_images_training = Minput_images[train_indices]
else:
    Minput_images_test = None
    Minput_images_training = None

if Minput_images_polar is not None:
    Minput_images_polar_test = Minput_images_polar[test_indices]
    Minput_images_polar_training = Minput_images_polar[train_indices]
else:
    Minput_images_polar_test = None
    Minput_images_polar_training = None

# === Affichage d'une partie du Minput_training et Moutput_training ===
print("\n" + "="*80)
print("Caractéristiques de la matrice Minput_training :")
print(f"- Shape (lignes, colonnes) : {Minput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_training[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_training :")
print(f"- Shape (lignes, colonnes) : {Moutput_training.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_training[:, :5].astype(int))  

# === Affichage d'une partie du Minput_test et Moutput_test ===
print("\n" + "="*80)
print("Caractéristiques de la matrice Minput_test :")
print(f"- Shape (lignes, colonnes) : {Minput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput_test[:, :5], 4))  

print("\nCaractéristiques de la matrice Moutput_test :")
print(f"- Shape (lignes, colonnes) : {Moutput_test.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput_test[:, :5].astype(int))  

# === Enregistrement en format .npy ===
print("\n" + "="*80)
print("Saving .npy files...")

np.save(os.path.join(script_dir, 'Minput_training.npy'), Minput_training)
np.save(os.path.join(script_dir, 'Moutput_training.npy'), Moutput_training)
np.save(os.path.join(script_dir, 'Minput_test.npy'), Minput_test)
np.save(os.path.join(script_dir, 'Moutput_test.npy'), Moutput_test)

if Minput_images_training is not None:
    np.save(os.path.join(script_dir, 'Minput_images_training.npy'), Minput_images_training)
    print(f"✓ Saved Minput_images_training.npy: shape {Minput_images_training.shape}")

if Minput_images_test is not None:
    np.save(os.path.join(script_dir, 'Minput_images_test.npy'), Minput_images_test)
    print(f"✓ Saved Minput_images_test.npy: shape {Minput_images_test.shape}")

if Minput_images_polar_training is not None:
    np.save(os.path.join(script_dir, 'Minput_images_polar_training.npy'), Minput_images_polar_training)
    print(f"✓ Saved Minput_images_polar_training.npy: shape {Minput_images_polar_training.shape}")

if Minput_images_polar_test is not None:
    np.save(os.path.join(script_dir, 'Minput_images_polar_test.npy'), Minput_images_polar_test)
    print(f"✓ Saved Minput_images_polar_test.npy: shape {Minput_images_polar_test.shape}")

# === Enregistrement en format .mat (optional) ===
print("\nSaving .mat files...")

scipy.io.savemat(os.path.join(script_dir, 'Minput_training.mat'), {'Minput_training': Minput_training})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_training.mat'), {'Moutput_training': Moutput_training})
scipy.io.savemat(os.path.join(script_dir, 'Minput_test.mat'), {'Minput_test': Minput_test})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_test.mat'), {'Moutput_test': Moutput_test})

print("✓ .mat files saved")

print("\n" + "="*80)
print("✅ Repartition complete! Training/test files ready for next step.")
