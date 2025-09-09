**Getting Started**
**Installation**
•	The code requires `python>=3.7`, as well as PyTorch and TorchVision. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

•	The following part gives more guidance for the installation and conda environment creation.

conda create -n 3d_gaze python=3.7.1

• To install required python library in your conda environment run the command
pip install -r requirements.txt

**Dataset Preparation**
•	First download the code, and the dataset.

•	After downloading the dataset, the data generation process should be completed.

•	After downloading the dataset keep the videos in Datasets/TEyeD/TEyeDSSingleFiles/Dikablis/VIDEOS/ and
 keep the annotations in  Datasets/TEyeD/TEyeDSSingleFiles/Dikablis/ANNOTATIONS/

 To produce the results, I have provided the Datasets folder which contains the dataset set videos and annotations. you can download Datasets folder from this link and use it in code. Link: https://mega.nz/folder/CQ91hKoI#5VJAFykpGniuJoxuZD6lNQ
 
•	To extract the necessary images for the training and testing for the video recording the Python file /data_generation/Extract_TEyeD.py should be run.

•	After processing the recordings, the data split has to be performed. The specifications of the recording's names that are used for training and testing are mentioned in the /cur_objs/datasetSelections.py. It will save datasetSelections.pkl file

•	To generate the training, validation and testing dataset .pkl file you need to run the following python file cur_objs/createDataloaders_baseline.py. It will save cond_TEyeD.pkl file in cur_objs/one_vs_one/. This pkl file contains splitting of the dataset in training, validation and testing dataset

•	To run createDataloaders_baseline.py  run the code

python createDataloaders_baseline.py \
--path2data “path_to_Datasets_folder” \
--ratios "0.7,0.15,0.15" \
--seed 42 \
--out "cur_objs/one_vs_one/cond_TEyeD.pkl"

•	Finally, you can customize and change arguments in args_maker.py file, like the optimizer, learning rates, weight decay, weight for losses, activate different heads (segmentation or rendering) etc.

**Training run script:**

•	To be able to run and train code the following command must to executed. The entry script is the run.py.

python -u run.py \
--exp_name='3D_eye_framework' \
--path_data=". /Datasets/All" \
--path2MasterKey=". /Datasets/MasterKey" \
--path_exp_tree=". /Results" \
--cur_obj='TEyeD' \
--use_pkl_for_dataload True \
--train_data_percentage=1.0 \
--random_dataloader \
--aug_flag=1 \
--reduce_valid_samples=0 \
--workers=8 \
--remove_spikes=0 \
--epochs=2 \
--batches_per_ep=4000 \
--lr=8e-3 \
--wd=2e-2 \
--batch_size=1 \
--frames=4 \
--early_stop_metric=3D \
--early_stop=20 \
--model='res_50_3' \
--net_rend_head \
--net_simply_head_tanh=1 \
--temp_n_angles=100 \
--temp_n_radius=50 \
--loss_w_ellseg=0.0 \
--loss_rend_vectorized \
--loss_w_rend_gt_2_pred=0.15 \
--loss_w_rend_pred_2_gt=0.15 \
--loss_w_rend_pred_2_gt_edge=0.0 \
--loss_w_rend_diameter=0.0 \
--loss_w_supervise=1 \
--loss_w_supervise_gaze_vector_3D_L2=2.5 \
--loss_w_supervise_gaze_vector_3D_cos_sim=2.5 \
--loss_w_supervise_gaze_vector_UV=0.0 \
--loss_w_supervise_eyeball_center=0.15 \
--loss_w_supervise_pupil_center=0.0 \
--do_distributed=0 \
--local_rank=0 \
--use_GPU=1 \
--mode='one_vs_one' \
--dropout=0 \
--use_ada_instance_norm_mixup=0

•	After running this script the .pth file is saved in Results/results folder.

**Testing run script:**

•	The check point given by the Author is stored at Results/pretrain_sem/results/last.pt

I have provoded that pth file. you can download it and use it in code. Link: https://drive.google.com/file/d/1XznFa7kyQvzoAZwo3pgMCNcXVfFOmnZR/view?usp=sharing

•	To be able to run and test code the following command must to executed. The entry script is the run.py.

python -u run.py \
--path_data=". /Datasets/All" \  
--path2MasterKey=". /Datasets/MasterKey" \  
--path_exp_tree=". /Results" \  
--use_pkl_for_dataload True \  
--only_test 1 \   
--weights_path=".Results/pretrained_sem/results/last.pt" \  
--model="res_50_3"\   
--exp_name="pretrained_sem"  \  
--use_GPU=1 \   
--net_rend_head \   
--loss_w_rend_pred_2_gt_edge 1 \   
--batches_per_ep 50 \   
--save_test_maps \

•	After running this code it will save test_results.h5 file in Results/pretrained_sem/results folder

**Visualization on Testing Data:**

•	To be able to run and visualization code the following command must to executed. 

python viz_from_testh5.py \
  --test_h5 ". /Results/pretrained_sem/results/test_results.h5" \
  --out_dir " ./Results/pretrained_sem/results/visual" 
  
  
•	It will save the visual grid in  /Results/pretrained_sem/results/visual folder.
•	The visulization of eye with predicted mask and gaze vector is stored in Results/pretrained_sem/results/Visual folder. Also the metrics are stored in Results/pretrained_sem/results folder in  test_results_summary.json file
<img width="3164" height="5237" alt="qual_grid_32_pred_pred" src="https://github.com/user-attachments/assets/09862a06-ef2e-41ea-bd8e-3baef51f1cfa" />




