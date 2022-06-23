import os
import shutil
import cc3d
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
            #extract te subjects using the adc file
            split_subjects = [f for f in os.listdir(os.path.join(path_to_dataset, center, split)) if "adc." in f]
            for subject in split_subjects:
                source_file = os.path.join(path_mask_archive, subject.replace("_adc.", "_ANTS_HDBET_FAST_smooth_concomp_defuseSeg."))
                if os.path.exists(source_file):
                    #overwrite the existing mask with the new mask
                    new_mask = nb.load(source_file)
                    affine = new_mask.affine
                    new_mask = new_mask.get_fdata()[2:146, 2:146, -42:]#same redim as applied to the adc and orginal mask
                    nb.save(nb.Nifti1Image(new_mask, affine), os.path.join(path_to_dataset, center, split, subject.replace("_adc.","_msk.")))
                    """
                    shutil.copy(source_file,
                                os.path.join(path_to_dataset, center, split, subject))
                    """
                else:
                    count_missing+=1
                    print("new label does not exist for", subject, "keeping the old labels!")

    print(count_missing, "subjects could not have updates with the new labels")

def adding_modalities(path_to_dataset, path_to_modality, name_modality):
    """ Replace one dataset mask with new masks, using the name of the subject
    """
    count_missing = 0
    for center in os.listdir(path_to_dataset):
        print("processing", str(center))
        for split in ["train", "valid", "test"]:
            #extract the subjects using the mask file
            split_subjects = [f for f in os.listdir(os.path.join(path_to_dataset, center, split)) if "msk." in f]
            for subject in split_subjects:
                shutil.copy(os.path.join(path_to_modality, center, split, subject.replace("msk.","adc.")),
                            os.path.join(path_to_dataset, center, split, subject.replace("msk.", name_modality+".")))

    print(count_missing, "subjects could not have updates with the new labels")

def apply_cc3d(path_to_dataset, connectivity=26):
    for center in os.listdir(path_to_dataset):
        print("processing", str(center))
        for split in ["train", "valid", "test"]:
            tmp_path = os.path.join(path_to_dataset, center, split)
            #extract the subjects using the mask file
            split_subjects = [f for f in os.listdir(tmp_path) if "msk." in f]
            for subject in split_subjects:
                bin_msk = nb.load(os.path.join(tmp_path, subject))
                msk_affine = bin_msk.affine
                #should maybe remove the blob that are smaller than XX to prevent having > 20 unique single components
                lbl_msk = cc3d.connected_components(bin_msk.get_fdata(), connectivity=connectivity) #face, vertex or corner adjacent by default
                nb.save(nb.Nifti1Image(lbl_msk, msk_affine), os.path.join(path_to_dataset, center, split, subject.replace("_msk.","_msk_labeled.")))