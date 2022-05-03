import os
import warnings
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
#parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')
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

warnings.filterwarnings("ignore")
Schedule = []

DR_LOCALIZATION = ['55']
METHODS  = ['DeepLab']
DA_TYPES = ['DR']
REFERENCES = ['None']#'REFERENCE_2017_EPSG4674_R225_62_CVA_OTSU_COS_SIM_Mrg_0_Nsy_0_PRef_0_Met_P-77R-43F1-55']
Dataset_MAIN_PATH = '/datawork/DATA/CHANGE_DETECTION/'
Checkpoint_Results_MAIN_PATH = '/datawork/EXPERIMENTS/Domain_Adaptation/'

for dr_localization in DR_LOCALIZATION:
    for method in METHODS:
        for da in DA_TYPES:
            for reference in REFERENCES:
                # Mudei o overlap para 0.2 e as epochs para 2
                #Tr: RO->PA
                Schedule.append("python " + Train_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --DR_Localization " + dr_localization + " --skip_connections False --epochs 100 --batch_size 32 --lr 0.0001 "
                                "--beta1 0.9 --data_augmentation True --source_vertical_blocks 10 --source_horizontal_blocks 10 --target_vertical_blocks 5 --target_horizontal_blocks 3 "
                                "--fixed_tiles True --defined_before False --image_channels 7 --patches_dimension 64 "
                                "--overlap_s 0.9 --overlap_t 0.9 --compute_ndvi False --balanced_tr True "
                                "--buffer True --source_buffer_dimension_out 2 --source_buffer_dimension_in 0 --target_buffer_dimension_out 2 --target_buffer_dimension_in 0 --porcent_of_last_reference_in_actual_reference 100 "
                                "--porcent_of_positive_pixels_in_actual_reference_s 2 --porcent_of_positive_pixels_in_actual_reference_t 2 "
                                "--num_classes 2 --phase train --training_type domain_adaptation --da_type " + da + " --runs 5 "
                                "--patience 10 --checkpoint_dir checkpoint_tr_AMAZON_RO_to_AMAZON_PA_" + method + "_" + da + "_" + reference + " "
                                "--source_dataset Amazon_RO --target_dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --source_data_t1_year 2016 --source_data_t2_year 2017 --target_data_t1_year 2016 --target_data_t2_year 2017 "
                                "--source_data_t1_name 18_07_2016_image_R232_67 --source_data_t2_name 21_07_2017_image_R232_67 --target_data_t1_name 02_08_2016_image_R225_62 --target_data_t2_name 20_07_2017_image_R225_62 "
                                "--source_reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --source_reference_t2_name REFERENCE_2017_EPSG32620_R232_67 --target_reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62 --target_reference_t2_name " + reference  + " "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

                #Tr: RO, Ts: PA->RO
                Schedule.append("python " + Test_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --DR_Localization " + dr_localization + " --skip_connections False --batch_size 500 --vertical_blocks 5 "
                                "--horizontal_blocks 3 --overlap 0.75 --image_channels 7 --patches_dimension 64 --compute_ndvi False --num_classes 2 "
                                "--phase test --training_type domain_adaptation --da_type " + da + " --checkpoint_dir checkpoint_tr_AMAZON_RO_to_AMAZON_PA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_AMAZON_PA_ts_AMAZON_PA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

                Schedule.append("python " + Metrics_05_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --skip_connections False --vertical_blocks 5 "
                                "--horizontal_blocks 3 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 "
                                "--compute_ndvi False --phase compute_metrics --training_type classification "
                                "--save_result_text True --checkpoint_dir checkpoint_tr_AMAZON_RO_to_AMAZON_PA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_AMAZON_PA_ts_AMAZON_PA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

                Schedule.append("python " + Metrics_th_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --skip_connections False --vertical_blocks 5 "
                                "--horizontal_blocks 3 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 --Npoints 100 "
                                "--compute_ndvi False --phase compute_metrics --training_type classification "
                                "--save_result_text False --checkpoint_dir checkpoint_tr_AMAZON_RO_to_AMAZON_PA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_AMAZON_PA_ts_AMAZON_PA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                                "--reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")



REFERENCES = ['None']#'REFERENCE_2018_EPSG4674_R220_63_CVA_OTSU_COS_SIM_Mrg_0_Nsy_0_PRef_0_Met_P-83R-65F1-73']

for dr_localization in DR_LOCALIZATION:
    for method in METHODS:
        for da in DA_TYPES:
            for reference in REFERENCES:
                #Adaptation
                #Tr: RO->MA
                Schedule.append("python " + Train_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --DR_Localization " + dr_localization + " --skip_connections False --epochs 100 --batch_size 32 --lr 0.0001 "
                                "--beta1 0.9 --data_augmentation True --source_vertical_blocks 10 --source_horizontal_blocks 10 --target_vertical_blocks 3 --target_horizontal_blocks 5 "
                                "--fixed_tiles True --defined_before False --image_channels 7 --patches_dimension 64 "
                                "--overlap_s 0.9 --overlap_t 0.9 --compute_ndvi False --balanced_tr True "
                                "--buffer True --source_buffer_dimension_out 2 --source_buffer_dimension_in 0 --target_buffer_dimension_out 2 --target_buffer_dimension_in 0 --porcent_of_last_reference_in_actual_reference 100 "
                                "--porcent_of_positive_pixels_in_actual_reference_s 2 --porcent_of_positive_pixels_in_actual_reference_t 2 "
                                "--num_classes 2 --phase train --training_type domain_adaptation --da_type " + da + " --runs 5 "
                                "--patience 10 --checkpoint_dir checkpoint_tr_AMAZON_RO_to_CERRADO_MA_" + method + "_" + da + "_" + reference + " "
                                "--source_dataset Amazon_RO --target_dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --source_data_t1_year 2016 --source_data_t2_year 2017 --target_data_t1_year 2017 --target_data_t2_year 2018 "
                                "--source_data_t1_name 18_07_2016_image_R232_67 --source_data_t2_name 21_07_2017_image_R232_67 --target_data_t1_name 18_08_2017_image_R220_63 --target_data_t2_name 21_08_2018_image_R220_63 "
                                "--source_reference_t1_name PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67 --source_reference_t2_name REFERENCE_2017_EPSG32620_R232_67 --target_reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63 --target_reference_t2_name " + reference  + " "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")
                #Tr: RO, Ts: MA-RO
                Schedule.append("python " + Test_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --DR_Localization " + dr_localization + " --skip_connections False --batch_size 500 --vertical_blocks 3 "
                                "--horizontal_blocks 5 --overlap 0.75 --image_channels 7 --patches_dimension 64 --compute_ndvi False --num_classes 2 "
                                "--phase test --training_type domain_adaptation --da_type " + da + " --checkpoint_dir checkpoint_tr_AMAZON_RO_to_CERRADO_MA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_CERRADO_MA_ts_CERRADO_MA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                                "--data_t1_name 18_08_2017_image_R220_63 --data_t2_name 21_08_2018_image_R220_63 "
                                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63 --reference_t2_name REFERENCE_2018_EPSG4674_R220_63 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

                Schedule.append("python " + Metrics_05_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --skip_connections False --vertical_blocks 3 "
                                "--horizontal_blocks 5 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 "
                                "--compute_ndvi False --phase compute_metrics --training_type classification "
                                "--save_result_text True --checkpoint_dir checkpoint_tr_AMAZON_RO_to_CERRADO_MA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_CERRADO_MA_ts_CERRADO_MA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                                "--data_t1_name 18_08_2017_image_R220_63 --data_t2_name 21_08_2018_image_R220_63 "
                                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63 --reference_t2_name REFERENCE_2018_EPSG4674_R220_63 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")

                Schedule.append("python " + Metrics_th_MAIN_COMMAND + " --classifier_type " + method + " --domain_regressor_type FC --skip_connections False --vertical_blocks 3 "
                                "--horizontal_blocks 5 --patches_dimension 64 --fixed_tiles True --overlap 0.75 --buffer True "
                                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 --Npoints 100 "
                                "--compute_ndvi False --phase compute_metrics --training_type classification "
                                "--save_result_text False --checkpoint_dir checkpoint_tr_AMAZON_RO_to_CERRADO_MA_" + method + "_" + da + "_" + reference + " --results_dir results_tr_AMAZON_RO_to_CERRADO_MA_ts_CERRADO_MA_" + method + "_" + da + "_" + reference + " "
                                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                                "--data_t1_name 18_08_2017_image_R220_63 --data_t2_name 21_08_2018_image_R220_63 "
                                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674_R220_63 --reference_t2_name REFERENCE_2018_EPSG4674_R220_63 "
                                "--dataset_main_path "+ Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path "+ Checkpoint_Results_MAIN_PATH + "")



for i in range(len(Schedule)):
    os.system(Schedule[i])
