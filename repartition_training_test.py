import numpy as np
import scipy.io
from scipy.io import loadmat, savemat
import os
import platform
import sys

np.random.seed(46) 

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path_input = os.path.join(script_dir, 'Minput.npy')
Minput = np.load(file_path_input)

file_path_output = os.path.join(script_dir, 'Moutput.npy')
Moutput = np.load(file_path_output)

try:
    file_path_images = os.path.join(script_dir, 'Minput_images.npy')
    Minput_images = np.load(file_path_images)
except FileNotFoundError:
    Minput_images = None

try:
    file_path_images_polar = os.path.join(script_dir, 'Minput_images_polar.npy')
    Minput_images_polar = np.load(file_path_images_polar)
except FileNotFoundError:
    Minput_images_polar = None

total_samples = Minput.shape[1]
rand_indices = np.random.permutation(total_samples)

nb_test = round(0.15 * total_samples)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

print(f"Total samples: {total_samples}")
print(f"Training samples: {len(train_indices)}")
print(f"Test samples: {len(test_indices)}")

Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]

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

np.save(os.path.join(script_dir, 'Minput_training.npy'), Minput_training)
np.save(os.path.join(script_dir, 'Moutput_training.npy'), Moutput_training)
np.save(os.path.join(script_dir, 'Minput_test.npy'), Minput_test)
np.save(os.path.join(script_dir, 'Moutput_test.npy'), Moutput_test)

if Minput_images_training is not None:
    np.save(os.path.join(script_dir, 'Minput_images_training.npy'), Minput_images_training)

if Minput_images_test is not None:
    np.save(os.path.join(script_dir, 'Minput_images_test.npy'), Minput_images_test)

if Minput_images_polar_training is not None:
    np.save(os.path.join(script_dir, 'Minput_images_polar_training.npy'), Minput_images_polar_training)

if Minput_images_polar_test is not None:
    np.save(os.path.join(script_dir, 'Minput_images_polar_test.npy'), Minput_images_polar_test)

scipy.io.savemat(os.path.join(script_dir, 'Minput_training.mat'), {'Minput_training': Minput_training})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_training.mat'), {'Moutput_training': Moutput_training})
scipy.io.savemat(os.path.join(script_dir, 'Minput_test.mat'), {'Minput_test': Minput_test})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_test.mat'), {'Moutput_test': Moutput_test})

print("Repartition complete! Training/test files ready for next step.")
