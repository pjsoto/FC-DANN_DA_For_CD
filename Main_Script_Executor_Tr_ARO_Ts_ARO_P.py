import os
import warnings
import argparse

warnings.filterwarnings("ignore")
Schedule = []
parser = argparse.ArgumentParser(description='')
parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_PBS', help='Decide wether the script will be running')
args = parser.parse_args()

if args.running_in == 'Datarmor_Interactive':
    Train_MAIN_COMMAND = "Main_Train_FC114.py"
    Test_MAIN_COMMAND = "Main_Test_FC114.py"
    Metrics_05_MAIN_COMMAND = "Main_Compute_Metrics_05.py"
    Metrics_th_MAIN_COMMAND = "Main_Compute_Average_Metrics_MT.py"
if args.running_in == 'Datarmor_PBS':
    Train_MAIN_COMMAND = "$HOME/CODE/CHANGE_DETECTION/FC-DANN_DA_For_CD/Main_Train_FC114.py"
    Test_MAIN_COMMAND = "$HOME/CODE/CHANGE_DETECTION/FC-DANN_DA_For_CD/Main_Test_FC114.py"
    Metrics_05_MAIN_COMMAND = "$HOME/CODE/CHANGE_DETECTION/FC-DANN_DA_For_CD/Main_Compute_Metrics_05.py"
    Metrics_th_MAIN_COMMAND = "$HOME/CODE/CHANGE_DETECTION/FC-DANN_DA_For_CD/Main_Compute_Average_Metrics_MT.py"
if args.running_in == 'Local_docker':
    Train_MAIN_COMMAND = "Main_Train_FC114.py"
    Test_MAIN_COMMAND = "Main_Test_FC114.py"
    Metrics_05_MAIN_COMMAND = "Main_Compute_Metrics_05.py"
    Metrics_th_MAIN_COMMAND = "Main_Compute_Average_Metrics_MT.py"

METHODS  = ['DeepLab']
DR_LOCALIZATION = ['55']
DA_TYPES = ['None']
REFERENCES = ['REFERENCE_2017_EPSG32620_R232_67_CVA_OTSU_COS_SIM_Mrg_0_Nsy_0_PRef_0_Met_P-47R-37F1-42']
Dataset_MAIN_PATH = '/datawork/DATA/CHANGE_DETECTION/'
Checkpoint_Results_MAIN_PATH = '/datawork/EXPERIMENTS/Domain_Adaptation/'

#Tr: PA, Ts: PA Pseudo Labels
Schedule.append("python " + Train_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --DR_Localization " + DR_LOCALIZATION[0] + " --skip_connections False --epochs 100 --batch_size 32 --lr 0.0001 "
                "--beta1 0.9 --data_augmentation True --source_vertical_blocks 10 --source_horizontal_blocks 10 --target_vertical_blocks 10 --target_horizontal_blocks 10 "
                "--fixed_tiles True --defined_before False --image_channels 7 --patches_dimension 64 "
                "--overlap_s 0.9 --overlap_t 0.9 --compute_ndvi False --balanced_tr True "
                "--buffer True --source_buffer_dimension_out 2 --source_buffer_dimension_in 0 --target_buffer_dimension_out 2 --target_buffer_dimension_in 0 --porcent_of_last_reference_in_actual_reference 100 "
                "--porcent_of_positive_pixels_in_actual_reference_s 2 --porcent_of_positive_pixels_in_actual_reference_t 2 "
                "--num_classes 2 --phase train --training_type classification --da_type " + DA_TYPES[0] + " --runs 5 "
                "--patience 10 --checkpoint_dir checkpoint_tr_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " "
                "--source_dataset Amazon_RO --target_dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --source_data_t1_year 2017 --source_data_t2_year 2018 --target_data_t1_year 2016 --target_data_t2_year 2017 "
                "--source_data_t1_name 18_07_2016_image_R232_67 --source_data_t2_name 21_07_2017_image_R232_67 --target_data_t1_name 18_07_2016_image_R232_67 --target_data_t2_name 21_07_2017_image_R232_67 "
                "--source_reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --source_reference_t2_name  " + REFERENCE[0]  + " --target_reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --target_reference_t2_name " + REFERENCE[0]  + " "
                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")


Schedule.append("python " + Test_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --DR_Localization " + DR_LOCALIZATION[0] + " --skip_connections False --batch_size 500 --vertical_blocks 10 "
                "--horizontal_blocks 10 --overlap 0.75 --image_channels 7 --patches_dimension 64 --compute_ndvi False --num_classes 2 "
                "--phase test --training_type classification --da_type " + DA_TYPES[0] + " --checkpoint_dir checkpoint_tr_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " --results_dir results_tr_AMAZON_RO_ts_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image_R232_67 --data_t2_name 21_07_2017_image_R232_67 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620_R232_67 "
                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

Schedule.append("python " + Metrics_05_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --skip_connections False --vertical_blocks 10 "
                "--horizontal_blocks 10 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text True --checkpoint_dir checkpoint_tr_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " --results_dir results_tr_AMAZON_RO_ts_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image_R232_67 --data_t2_name 21_07_2017_image_R232_67 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620_R232_67 "
                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

Schedule.append("python " + Metrics_th_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --skip_connections False --vertical_blocks 10 "
                "--horizontal_blocks 10 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 --Npoints 100 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text False --checkpoint_dir checkpoint_tr_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " --results_dir results_tr_AMAZON_RO_ts_AMAZON_RO_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + REFERENCE[0] + " "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image_R232_67 --data_t2_name 21_07_2017_image_R232_67 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620_R232_67 "
                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

for i in range(len(Schedule)):
    os.system(Schedule[i])
