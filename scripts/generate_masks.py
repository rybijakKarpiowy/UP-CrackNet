import os
import numpy as np
from matplotlib import pyplot as plt

masks_dir = './ck_mask/'

# Generate 6 masks of size 448x448 and save as .npy files
for size in range(4):
    for black_first in [True, False]:
        squares_num = 2 ** (size + 1)
        cbs = 448 // squares_num  # cell block size
        mask = np.zeros((448, 448, 3), dtype=np.uint8)
        for i in range(squares_num):
            for j in range(squares_num):
                if (i + j) % 2 == 0 and black_first:
                    mask[i * cbs:(i + 1) * cbs, j * cbs:(j + 1) * cbs] = 1
                elif (i + j) % 2 == 1 and not black_first:
                    mask[i * cbs:(i + 1) * cbs, j * cbs:(j + 1) * cbs] = 1
        mask_filename = f'ck_{size * 2 + (0 if black_first else 1)}.npy'
        np.save(os.path.join(masks_dir, mask_filename), mask)
        print(f'Saved mask: {mask_filename}')
        
        # Optional: visualize the mask
        plt.imshow(mask[:, :, 0], cmap='gray')
        plt.title(f'Mask: {mask_filename}')
        plt.show()