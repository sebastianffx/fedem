import unittest
import torch
import torchio as tio

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from preprocessing import torchio_generate_loaders, torchio_create_test_transfo, \
                        brats_torchio_create_transform,isles22_torchio_create_transform

"""
class TestDatasetLoading(unittest.TestCase):
    #verify that the loading function are handling all the supported datasets hierarchy
    #load the dataset
    self.dataloaders, self.all_test_loader, self.all_valid_loader, self.all_train_loader = torchio_generate_loaders(partitions_paths=options["partitions_paths"],
                                                                                                                    batch_size=self.options["batch_size"],
                                                                                                                    clamp_min=self.options["clamp_min"],
                                                                                                                    clamp_max=self.options["clamp_max"],
                                                                                                                    padding=self.options["padding"],
                                                                                                                    patch_size=self.options["patch_size"],
                                                                                                                    max_queue_length=self.options["max_queue_length"],
                                                                                                                    patches_per_volume=self.options["patches_per_volume"],
                                                                                                                    no_deformation=self.options["no_deformation"],
                                                                                                                    partitions_paths_add_mod=self.options["partitions_paths_add_mod"]
                                                                                                                    )

    def test_partition_multisite(self):
        centers_partitions = get_train_valid_test_partitions(path, modality, clients, folder_struct="site_simple", multi_label=False, additional_modalities=[])
                                                                                                                            
        self.assertEqual('foo'.upper(), 'FOO')

    def test_partition_multisite_nested(self):
        centers_partitions = get_train_valid_test_partitions(path, modality, clients, folder_struct="site_nested", multi_label=False, additional_modalities=[])
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_partition_multisite_nested(self):
        centers_partitions = get_train_valid_test_partitions(path, modality, clients, folder_struct="folder_simple", multi_label=False, additional_modalities=[])
        
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
"""
class TestDataAugmentation(unittest.TestCase):
    """verify that the torchio transform function are behaving as expected
    """
    #create dummy inputs to test the torchio transformation functions
    def setUp(self):
        x = 6
        #3D cross with different values for each bar/branch
        self.cross_multi = torch.zeros((16,16,16))
        self.cross_multi[:, x:-x, x:-x] = 1
        self.cross_multi[x:-x, :, x:-x] = 2
        self.cross_multi[x:-x, x:-x, :] = 3
        self.cross_multi[x:-x, x:-x, x:-x] = 4
        self.cross_multi = self.cross_multi.unsqueeze(dim=0)
        #label = the intersection of the 6 branches, center of the cube
        self.cross_multi_label = self.cross_multi == 4

        #3D cross
        self.cross_simple = torch.zeros((16,16,16))
        self.cross_simple[:, x:-x, x:-x] = 1
        self.cross_simple[x:-x, :, x:-x] = 1
        self.cross_simple[x:-x, x:-x, :] = 1
        self.cross_simple = self.cross_simple.unsqueeze(dim=0)
        #label = the half of the branch touching the front face (from face to the center)
        self.cross_simple_label = torch.zeros((16,16,16))
        self.cross_simple_label[:x, x:-x, x:-x] = 1
        self.cross_simple_label = self.cross_simple_label.unsqueeze(dim=0)

        #cube with one corner filled with ones (front top left corner)
        self.cube_corner = torch.zeros((16,16,16))
        self.cube_corner[:8,:8,:8] = 1
        self.cube_corner = self.cube_corner.unsqueeze(dim=0)
        #label = itself, the corner filled with ones
        self.cube_corner_label = self.cube_corner.clone()

    # test individual transformation
    def test_single_transform(self):
        #min-max normalization
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        self.assertTrue(torch.allclose(rescale(self.cross_multi), (self.cross_multi-0)/(4-0)))
        self.assertTrue(torch.allclose(rescale(self.cross_simple), self.cross_simple))

        #rotations
        rotation = tio.Affine(scales=1, degrees=90, translation=0) #rotation of 90# on all axes
        self.assertTrue(torch.allclose(rotation(self.cross_simple), self.cross_simple))
        solution = torch.zeros((1,16,16,16))
        solution[:,8:,:8,:8] = 1
        #after 90° rotation on each axis, 1-corner is in the front bottom left corner
        self.assertTrue(torch.allclose(rotation(self.cube_corner), solution))

        #flipping
        flipping_right = tio.Flip(axes="R")
        solution = torch.zeros((1,16,16,16))
        solution[:,8:,:8,:8] = 1
        #right axis flipping, 1-corner is in the front bottom left corner
        self.assertTrue(torch.allclose(flipping_right(self.cube_corner), solution))

        flipping_posterior = tio.Flip(axes="P")
        solution = torch.zeros((1,16,16,16))
        solution[:,:8,8:,:8] = 1
        #posterior axis flipping, 1-corner is in the front top right corner
        self.assertTrue(torch.allclose(flipping_posterior(self.cube_corner), solution))

    def test_brats_transforms(self):

        subjects=[tio.Subject(feature_map=tio.ScalarImage(tensor=self.cross_multi),
                              label=tio.LabelMap(tensor=self.cross_multi_label),
                              name="complex_cross"),
                  tio.Subject(feature_map=tio.ScalarImage(tensor=self.cross_simple),
                              label=tio.LabelMap(tensor=self.cross_simple_label),
                              name="cross_branch"),
                  tio.Subject(feature_map=tio.ScalarImage(tensor=self.cross_simple),
                              label=tio.LabelMap(tensor=self.cross_simple),
                              name="simple_cross"),
                  tio.Subject(feature_map=tio.ScalarImage(tensor=self.cube_corner),
                              label=tio.LabelMap(tensor=self.cube_corner_label),
                              name="cube_corner")]

        train_transform, test_transform = brats_torchio_create_transform(padding=(4,4,2), patch_size=(8,8,4))

        train_loader = torch.utils.data.DataLoader(tio.SubjectsDataset(subjects, 
                                                                       transform=train_transform),
                                                   batch_size=1)

        for sample in train_loader:
            #check dimensions after padding
            self.assertEqual(sample["feature_map"]["data"].shape, (1,1,24,24,20)) #(16,16,16) + 2*(4,4,2)
            self.assertEqual(sample["label"]["data"].shape, (1,3,24,24,20)) #preprocessing pipeline create 3 labels channels
            #check normalization/scaling
            self.assertAlmostEqual(sample["feature_map"]["data"].max(), 1.)
            self.assertAlmostEqual(sample["feature_map"]["data"].min(), 0.)
            #check that labels was not scaled, the two possible values are 1 and 0
            self.assertEqual(len(torch.unique(sample["label"]["data"])), 2)

            #the label should still be aligned with the data!
            if sample["name"] in ["cube_corner", "simple_cross"]:
                self.assertTrue(torch.allclose(sample["feature_map"]["data"], sample["label"]["data"]))

    #check the validation transformation as well? Should check that to rotation/flipping is occuring?

    def test_isles_transforms(self):
        #must add channels to mimic the multi-channels = multi-modality approach + different sites scale
        subjects=[tio.Subject(ref_space=tio.ScalarImage(tensor=self.cross_multi),
                              feature_map=tio.ScalarImage(tensor=torch.stack([self.cross_multi*0.8, self.cross_multi], dim=0).squeeze()),
                              label=tio.LabelMap(tensor=self.cross_multi_label),
                              name="complex_cross"),
                  tio.Subject(ref_space=tio.ScalarImage(tensor=self.cross_simple),
                              feature_map=tio.ScalarImage(tensor=torch.stack([self.cross_simple*2, self.cross_simple*3], dim=0).squeeze()),
                              label=tio.LabelMap(tensor=self.cross_simple_label),
                              name="cross_branch"),
                  tio.Subject(ref_space=tio.ScalarImage(tensor=self.cross_simple),
                              feature_map=tio.ScalarImage(tensor=torch.stack([self.cross_simple*3e-3, self.cross_simple*4e-3], dim=0).squeeze()),
                              label=tio.LabelMap(tensor=self.cross_simple),
                              name="simple_cross"),
                  tio.Subject(ref_space=tio.ScalarImage(tensor=self.cube_corner),
                              feature_map=tio.ScalarImage(tensor=torch.stack([self.cube_corner*1e5, self.cube_corner*3e5], dim=1).squeeze()),
                              label=tio.LabelMap(tensor=self.cube_corner_label),
                              name="cube_corner")]

        train_transform, test_transform = isles22_torchio_create_transform(padding=(4,4,2), patch_size=(8,8,4), no_deformation=False)

        train_loader = torch.utils.data.DataLoader(tio.SubjectsDataset(subjects, 
                                                                       transform=train_transform),
                                                   batch_size=1)

        for sample in train_loader:
            #check dimensions after padding
            self.assertEqual(sample["feature_map"]["data"].shape, (1,2,24,24,20)) #(2,16,16,16) + 2*(0,4,4,2)
            self.assertEqual(sample["label"]["data"].shape, (1,1,24,24,20))
            #check normalization/scaling
            self.assertAlmostEqual(sample["feature_map"]["data"].max().item(), 1.)
            self.assertAlmostEqual(sample["feature_map"]["data"].min().item(), 0.)
            #check that labels was not scaled, the two possible values are 1 and 0
            self.assertEqual(len(torch.unique(sample["label"]["data"])), 2)

            #the label should still be aligned with the data!
            if sample["name"] in ["cube_corner", "simple_cross"]:
                self.assertTrue(torch.allclose(sample["feature_map"]["data"], sample["label"]["data"]))
        
if __name__ == '__main__':
    unittest.main(verbosity=2)

    
    
    