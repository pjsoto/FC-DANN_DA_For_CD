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
PSEUDLABELS_COEFFICIENTS = ['0.10','0.25','0.50','0.75','0.90','1.0']
REFERENCE = ['REFERENCE_2018_EPSG4674_R220_63_CVA_OTSU_PRIOR_0_COS_SIM_Mrg_0_Nsy_1_PRef_0_HConf_1_Met_P-83R-65F1-73']
PAST_REFERENCE = ['PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_CVA_OTSU_COS_SIM_Mrg_0_Nsy_1_PRef_0_HConf_1_Met_P-83R-65F1-73']

Dataset_MAIN_PATH = '/datawork/DATA/CHANGE_DETECTION/'
Checkpoint_Results_MAIN_PATH = '/datawork/EXPERIMENTS/Domain_Adaptation/'
for pseudo_labels_coefficient in PSEUDLABELS_COEFFICIENTS:
    #Tr: PA, Ts: PA Pseudo Labels
    Schedule.append("python " + Train_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --DR_Localization " + DR_LOCALIZATION[0] + " --skip_connections False --epochs 100 --batch_size 32 --lr 0.0001 "
                    "--beta1 0.9 --data_augmentation True --source_vertical_blocks 3 --source_horizontal_blocks 5 --target_vertical_blocks 3 --target_horizontal_blocks 5 "
                    "--fixed_tiles True --defined_before False --image_channels 7 --patches_dimension 64 "
                    "--overlap_s 0.9 --overlap_t 0.9 --compute_ndvi False --balanced_tr True "
                    "--buffer True --source_buffer_dimension_out 2 --source_buffer_dimension_in 0 --target_buffer_dimension_out 2 --target_buffer_dimension_in 0 --porcent_of_last_reference_in_actual_reference 100 "
                    "--porcent_of_positive_pixels_in_actual_reference_s 2 --porcent_of_positive_pixels_in_actual_reference_t 2 "
                    "--num_classes 2 --phase train --training_type classification --da_type " + DA_TYPES[0] + " --runs 5 --pseudo_labels_coefficient " + pseudo_labels_coefficient + " "
                    "--patience 10 --checkpoint_dir checkpoint_tr_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " "
                    "--source_dataset Cerrado_MA --target_dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                    "--data_type .npy --source_data_t1_year 2017 --source_data_t2_year 2018 --target_data_t1_year 2016 --target_data_t2_year 2017 "
                    "--source_data_t1_name 18_08_2017_image_R220_63_MA --source_data_t2_name 21_08_2018_image_R220_63_MA --target_data_t1_name 18_08_2017_image_R220_63_MA --target_data_t2_name 21_08_2018_image_R220_63_MA "
                    "--source_reference_t1_name " + PAST_REFERENCE[0] + " --source_reference_t2_name  " + REFERENCE[0]  + " --target_reference_t1_name " + PAST_REFERENCE[0] + " --target_reference_t2_name " + REFERENCE[0]  + " "
                    "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                    "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")


    Schedule.append("python " + Test_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --DR_Localization " + DR_LOCALIZATION[0] + " --skip_connections False --batch_size 500 --vertical_blocks 3 "
                    "--horizontal_blocks 5 --overlap 0.75 --image_channels 7 --patches_dimension 64 --compute_ndvi False --num_classes 2 "
                    "--phase test --training_type classification --da_type " + DA_TYPES[0] + " --checkpoint_dir checkpoint_tr_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " --results_dir results_tr_CERRADO_MA_ts_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " "
                    "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                    "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                    "--data_t1_name 18_08_2017_image_R220_63_MA --data_t2_name 21_08_2018_image_R220_63_MA "
                    "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_MA --reference_t2_name REFERENCE_2018_EPSG4674_R220_63_MA "
                    "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                    "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

    Schedule.append("python " + Metrics_05_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --skip_connections False --vertical_blocks 3 "
                    "--horizontal_blocks 5 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                    "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 "
                    "--compute_ndvi False --phase compute_metrics --training_type classification "
                    "--save_result_text True --checkpoint_dir checkpoint_tr_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " --results_dir results_tr_CERRADO_MA_ts_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " "
                    "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                    "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                    "--data_t1_name 18_08_2017_image_R220_63_MA --data_t2_name 21_08_2018_image_R220_63_MA "
                    "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_MA --reference_t2_name REFERENCE_2018_EPSG4674_R220_63_MA "
                    "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                    "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

    Schedule.append("python " + Metrics_th_MAIN_COMMAND + " --classifier_type " + METHODS[0] + " --domain_regressor_type None --skip_connections False --vertical_blocks 3 "
                    "--horizontal_blocks 5 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                    "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 --Npoints 100 "
                    "--compute_ndvi False --phase compute_metrics --training_type classification "
                    "--save_result_text False --checkpoint_dir checkpoint_tr_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " --results_dir results_tr_CERRADO_MA_ts_CERRADO_MA_" + METHODS[0] + "_" + DA_TYPES[0] + "_" + pseudo_labels_coefficient + "_" + REFERENCE[0] + " "
                    "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                    "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                    "--data_t1_name 18_08_2017_image_R220_63_MA --data_t2_name 21_08_2018_image_R220_63_MA "
                    "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_MA --reference_t2_name REFERENCE_2018_EPSG4674_R220_63_MA "
                    "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                    "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

for i in range(len(Schedule)):
    os.system(Schedule[i])
