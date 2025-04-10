import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import square, disk
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Tools import *

class AMAZON_RO():
    def __init__(self, args):

        self.images_norm = []
        self.references = []
        self.mask = []
        self.coordinates = []

        Image_t1_path = args.dataset_main_path +  "/AMAZON/IMAGES/" + args.data_t1_name + '.npy'
        Image_t2_path = args.dataset_main_path +  "/AMAZON/IMAGES/" + args.data_t2_name + '.npy'
        Reference_t1_path = args.dataset_main_path + "/AMAZON/REFERENCES/" + args.reference_t1_name + '.npy'
        Reference_t2_path = args.dataset_main_path + "/AMAZON/REFERENCES/" + args.reference_t2_name + '.npy'
        print(Image_t1_path)
        # Reading images and references
        print('[*]Reading images...')
        image_t1 = np.load(Image_t1_path)
        image_t2 = np.load(Image_t2_path)
        reference_t1 = np.load(Reference_t1_path)
        image_t1 = image_t1[:,1:2551,1:5121]
        image_t2 = image_t2[:,1:2551,1:5121]
        print(np.min(image_t1))
        print(np.max(image_t1))

        if reference_t1.shape[0] != image_t1.shape[1] or reference_t1.shape[1] != image_t1.shape[2]:
            reference_t1 = reference_t1[1:2551,1:5121]
        if os.path.exists(Reference_t2_path):
            reference_t2 = np.load(Reference_t2_path)
            if reference_t2.shape[0] != reference_t1.shape[0] or reference_t2.shape[1] != reference_t1.shape[1]:
                reference_t2 = reference_t2[1:2551,1:5121]
        elif args.reference_t2_name == 'None':
            reference_t2 = np.ones((2550, 5120))

        # Pre-processing references
        if args.buffer:
            print('[*]Computing buffer regions...')
            #Dilating the reference_t1
            reference_t1 = skimage.morphology.dilation(reference_t1, disk(args.buffer_dimension_out))
            if os.path.exists(Reference_t2_path) or args.reference_t2_name == 'NDVI':
                #Dilating the reference_t2
                reference_t2_dilated = skimage.morphology.dilation(reference_t2, disk(args.buffer_dimension_out))
                buffer_t2_from_dilation = reference_t2_dilated - reference_t2
                reference_t2_eroded  = skimage.morphology.erosion(reference_t2 , disk(args.buffer_dimension_in))
                buffer_t2_from_erosion  = reference_t2 - reference_t2_eroded
                buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
                reference_t2 = reference_t2 - buffer_t2_from_erosion
                buffer_t2[buffer_t2 == 1] = 2
                reference_t2 = reference_t2 + buffer_t2

        # Pre-processing images
        if args.compute_ndvi:
            print('[*]Computing and stacking the ndvi band...')
            ndvi_t1 = Compute_NDVI_Band(image_t1)
            ndvi_t2 = Compute_NDVI_Band(image_t2)
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
            image_t1 = np.concatenate((image_t1, ndvi_t1), axis=2)
            image_t2 = np.concatenate((image_t2, ndvi_t2), axis=2)
        else:
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))


        # Pre-Processing the images

        print('[*]Normalizing the images...')
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        images = np.concatenate((image_t1, image_t2), axis=2)
        images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))

        scaler = scaler.fit(images_reshaped)
        self.scaler = scaler
        images_normalized = scaler.fit_transform(images_reshaped)
        images_norm = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
        image_t1_norm = images_norm[:, :, : image_t1.shape[2]]
        image_t2_norm = images_norm[:, :, image_t2.shape[2]: ]

        print(np.min(image_t1_norm))
        print(np.max(image_t1_norm))
        print(np.min(image_t2_norm))
        print(np.max(image_t2_norm))

        # Storing the images in a list
        self.images_norm.append(image_t1_norm)
        self.images_norm.append(image_t2_norm)
        # Storing the references in a list
        self.references.append(reference_t1)
        self.references.append(reference_t2)

    def Tiles_Configuration(self, args, i):
        #Generating random training and validation tiles
        if args.phase == 'train' or args.phase == 'compute_metrics':
            if args.fixed_tiles:
                if args.defined_before:
                    if args.phase == 'train':
                        files = os.listdir(args.checkpoint_dir_posterior)
                        print(files[i])
                        self.Train_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Train_tiles.npy')
                        self.Valid_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Valid_tiles.npy')
                        np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                        np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                    if args.phase == 'compute_metrics':
                        self.Train_tiles = np.load(args.save_checkpoint_path +  'Train_tiles.npy')
                        self.Valid_tiles = np.load(args.save_checkpoint_path +  'Valid_tiles.npy')
                else:
                    self.Train_tiles = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
                    self.Valid_tiles = np.array([8, 11, 26, 49, 78])
                    self.Undesired_tiles = []
            else:
                tiles = np.random.randint(100, size = 25) + 1
                self.Train_tiles = tiles[:20]
                self.Valid_tiles = tiles[20:]
                np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
        if args.phase == 'test':
            self.Train_tiles = []
            self.Valid_tiles = []
            self.Undesired_tiles = []

    def Coordinates_Creator(self, args, i):
        self.images_norm_ = []
        self.references_ = []
        print('[*]Defining the central patches coordinates...')
        if args.phase == 'train':
            if args.fixed_tiles:
                if i == 0:
                    self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)

                self.corners_coordinates_tr, self.corners_coordinates_vl, reference1_, reference2_, self.pad_tuple, self.class_weights = Corner_Coordinates_Definition_Training(self.mask, self.references[0], self.references[1], args.patches_dimension, args.overlap, args.porcent_of_last_reference_in_actual_reference, args.porcent_of_positive_pixels_in_actual_reference)
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
            else:
                self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
                self.corners_coordinates_tr, self.corners_coordinates_vl, reference1_, reference2_, self.pad_tuple, self.class_weights = Corner_Coordinates_Definition_Training(self.mask, self.references[0], self.references[1], args.patches_dimension, args.overlap, args.porcent_of_last_reference_in_actual_reference, args.porcent_of_positive_pixels_in_actual_reference)

            self.references_.append(reference1_)
            self.references_.append(reference2_)
        if args.phase == 'test':
            self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
            self.corners_coordinates_ts, self.pad_tuple, self.k1, self.k2, self.step_row, self.step_col, self.stride, self.overlap = Corner_Coordinates_Definition_Testing(self.mask, args.patches_dimension, args.overlap)

        # Performing the corresponding padding into the images
        self.images_norm_.append(np.pad(self.images_norm[0], self.pad_tuple, mode='symmetric'))
        self.images_norm_.append(np.pad(self.images_norm[1], self.pad_tuple, mode='symmetric'))

        print(np.shape(self.images_norm))
