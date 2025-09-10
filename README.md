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
•	To set the hyperparameter and the required path for training config_train.py file is used. 
•	To be able to run and train code the following command must to executed. The entry script is the run.py.

python train.py

•	After running this script the .pth file is saved in Results/results folder.

**Testing run script:**
•	To set the hyperparameter and the required path for testing config_test.py file is used. 
•	The check point given by the Author should be stored at Results/pretrain_sem/results/last.pt 

I have provided that pth file given by the Author. you can download it and use it in code. Link: https://drive.google.com/file/d/1XznFa7kyQvzoAZwo3pgMCNcXVfFOmnZR/view?usp=sharing

•	To be able to run and test code the following command must to executed. The entry script is the run.py.

python test.py

•	After running this code it will save test_results.h5 file in Results/pretrained_sem/results folder

•	for testing the results, you can download the test_results.h5 file from this link: https://drive.google.com/file/d/1qL9nGS2zt-Zd6_UhnggsJfEIHu6CzOzF/view?usp=sharing


**Visualization on Testing Data:**

•	To be able to run and visualization code the following command must to executed. 

python visualization.py \
  --test_h5 ". /Results/pretrained_sem/results/test_results.h5" \
  --out_dir " ./Results/pretrained_sem/results/visual" 
  
  
•	It will save the visual grid in  /Results/pretrained_sem/results/visual folder.

•	The visulization of eye with predicted mask and gaze vector is stored in Results/pretrained_sem/results/Visual folder. Also the metrics are stored in Results/pretrained_sem/results folder in  test_results_summary.json file

<img width="3164" height="3916" alt="qual_grid_24_pred_pred" src="https://github.com/user-attachments/assets/4e0f52a8-96cf-4edd-b918-0fc2e8ea9b9f" />




