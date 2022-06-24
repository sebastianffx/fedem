import os
import shutil
import cc3d
import numpy as np
import pandas as pd
import nibabel as nb

from glob import glob

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

def get_same_res_paths(nifti_labl_paths, spacing=(2.0, 2.0, 2.0), folder_label="derivatives", folder_maps="rawdata_adc_transformed"):
    """ Take the path to the root of a dataset and share the data into two; one with identical spacing and another with the rest
    """
    labels_paths, other_labels_paths  = [],[]
    dwi_paths, other_dwi_paths   = [],[]
    flair_paths, other_flair_paths  = [],[]
    adc_paths, other_adc_paths  = [],[]
    for label_path in nifti_labl_paths:
        cur_nifti = nb.load(label_path)
        lbl_pixels = cur_nifti.get_fdata()
        sx, sy, sz = cur_nifti.header.get_zooms()
        volume = sx * sy * sz
        if (sx, sy, sz) == spacing : #maybe add some tolerances? In astral adata, there were variation of 1*e-4
            #save path to labels
            labels_paths.append(label_path)
            #switch to the modalities folder
            path_to_modality = label_path.replace(folder_label, folder_maps)
            flair_paths.append(path_to_modality.replace("_msk.nii.gz", "_flair.nii.gz"))
            dwi_paths.append(path_to_modality.replace("_msk.nii.gz", "_dwi.nii.gz"))
            adc_paths.append(path_to_modality.replace("_msk.nii.gz", "_adc.nii.gz"))
        else:
            #save path to labels
            other_labels_paths.append(label_path)
            #switch to the modalities folder
            path_to_modality = label_path.replace(folder_label, folder_maps)
            other_flair_paths.append(path_to_modality.replace("_msk.nii.gz", "_flair.nii.gz"))
            other_dwi_paths.append(path_to_modality.replace("_msk.nii.gz", "_dwi.nii.gz"))
            other_adc_paths.append(path_to_modality.replace("_msk.nii.gz", "_adc.nii.gz"))
    return labels_paths, dwi_paths, flair_paths, adc_paths, other_labels_paths, other_dwi_paths, other_flair_paths, other_adc_paths


def generateSPLIT(path_to_dataset, dataset_name, train_valid_test=[0.75,0.1,0.15], spacing=(2.0, 2.0, 2.0), gen_csv=False):

    nifti_labl_paths= glob(path_to_dataset)
   
    if gen_csv:
        lbl_paths, dwi_paths, flair_paths, adc_paths, o_lbl_paths, o_dwi_paths, o_flair_paths, o_adc_paths = get_same_res_paths(nifti_labl_paths,
                                                                                                                                    spacing=(2.0, 2.0, 2.0))
 
        print("WARNING, your generating a new subject split csv file")
        #shuffle based on labels, because modality agnostic
        percentages_train_val_test = [int(0.75*len(lbl_paths)),int(0.1*len(lbl_paths)),int(0.15*len(lbl_paths))]
        #rounding issues are going to the training set
        percentages_train_val_test[0] += len(lbl_paths) - np.sum(percentages_train_val_test)
        print("Train, Val, Test Imgs: ", percentages_train_val_test)
        assert len(lbl_paths) == np.sum(percentages_train_val_test)

        patients = [lbl_path.split("/")[-1] for lbl_path in lbl_paths]

        #generate csv file containing the informations for FEDEM training
        indexes=list(range(len(lbl_paths)))
        np.random.shuffle(indexes)
        df_fedem = pd.DataFrame(dict(zip(["label", "dwi", "flair", "adc"], [lbl_paths, dwi_paths, flair_paths, adc_paths])))
        df_fedem["site1"] = "train"
        df_fedem.loc[indexes[:percentages_train_val_test[1]], "site1"] = "valid"
        df_fedem.loc[indexes[percentages_train_val_test[1]:percentages_train_val_test[2]+percentages_train_val_test[1]], "site1"] = "test"
        df_fedem.to_csv("fedem_split_"+dataset_name+".csv")
    else:
        df_fedem = pd.read_csv("fedem_split_"+dataset_name+".csv", index_col=0)

    #COPIED from MASiVAR_ProcessingPipeline.py and adapted for multi-modality dataset
    nb_sites = len(df_fedem.columns)-1
    #master folder
    os.makedirs(dataset_name, exist_ok=True)
    for center_num in range(1, nb_sites+1):
        #site main folder
        os.makedirs(os.path.join(dataset_name, "center"+str(center_num)), exist_ok=True)
        target_folder = os.path.join(dataset_name, "center"+str(center_num))
        os.makedirs(os.path.join(target_folder, "train"), exist_ok=True)

        for f in df.loc[df["site"+str(center_num) == "train"]].iterrows():
            shutil.copy(f["label"],
                        os.path.join(target_folder, "train", f.split("/")[-1]))
            shutil.copy(f["dwi"],
                        os.path.join(target_folder, "train", f.split("/")[-1]))
            shutil.copy(f["flair"],
                        os.path.join(target_folder, "train", f.split("/")[-1]))
            shutil.copy(f["adc"],
                        os.path.join(target_folder, "train", f.split("/")[-1]))

        os.makedirs(os.path.join(target_folder, "valid"), exist_ok=True)
        for f in df.loc[df["site"+str(center_num) == "valid"]].iterrows():
            shutil.copy(f["label"],
                        os.path.join(target_folder, "valid", f.split("/")[-1]))
            shutil.copy(f["dwi"],
                        os.path.join(target_folder, "valid", f.split("/")[-1]))
            shutil.copy(f["flair"],
                        os.path.join(target_folder, "valid", f.split("/")[-1]))
            shutil.copy(f["adc"],
                        os.path.join(target_folder, "valid", f.split("/")[-1]))

        os.makedirs(os.path.join(target_folder, "test") , exist_ok=True)
        for f in df.loc[df["site"+str(center_num) == "test"]].iterrows():
            shutil.copy(f["label"],
                        os.path.join(target_folder, "test", f.split("/")[-1]))
            shutil.copy(f["dwi"],
                        os.path.join(target_folder, "test", f.split("/")[-1]))
            shutil.copy(f["flair"],
                        os.path.join(target_folder, "test", f.split("/")[-1]))
            shutil.copy(f["adc"],
                        os.path.join(target_folder, "test", f.split("/")[-1]))