import os
import shutil
import numpy as np
import nibabel as nb

def crop2seg(path, r_low, r_high):
    """ Crop volume and the corresponding labels to a given slices range
    """
    img = nb.load(path)
    img_affine = img.affine

    img = img.get_fdata()[:,:,r_low:r_high]
    nb.save(nb.Nifti1Image(img, img_affine), path)

    img = nb.load(path.replace("adc.", "msk."))
    img = img.get_fdata()[:,:,r_low:r_high]
    nb.save(nb.Nifti1Image(img, img_affine), path.replace("adc.", "msk."))

def replace_masks(path_to_dataset, path_mask_archive):
    """ Replace one dataset mask with new masks, using the name of the subject
    """
    #TODO: also copy the labels with component labeling, could be useful for the blob loss

    count_missing = 0
    for center in os.listdir(path_to_dataset):
        print("processing", str(center))
        for split in ["train", "valid", "test"]:
            #extract te subjects using the mask file
            split_subjects = [f for f in os.listdir(os.path.join(path_to_dataset, center, split)) if "msk." in f]
            for subject in split_subjects:
                source_file = os.path.join(path_mask_archive, subject.replace("_msk.", "_ANTS_HDBET_FAST_smooth_concomp_defuseSeg_labeled."))
                if os.path.exists(source_file):
                    #overwrite the existing mask with the new mask
                    shutil.copy(source_file,
                                os.path.join(os.path.join(path_to_dataset, center, split, subject)))
                else:
                    count_missing+=1
                    print("new label does not exist for", subject, "keeping the old labels!")

    print(count_missing, "subjects could not have updates with the new labels")