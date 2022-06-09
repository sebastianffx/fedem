import numpy as np
import nibabel as nb

def crop2seg(path, r_low, r_high):
    img = nb.load(path)
    img_affine = img.affine

    img = img.get_fdata()[:,:,r_low:r_high]
    nb.save(nb.Nifti1Image(img, img_affine), path)
