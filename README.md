# HECKTOR2022 (Team AIMERS)
<b>Head and neck tumor and lymph node auto-segmentation for PET/CT </b>


Created by AIM Lab members: Arnav Jain, Julia Huang, Yashwanth Ravipati, Gregory Cain, Aidan Boyd, Zezhong Ye and Benjamin Kann.

<b>Keywords:</b> Segmentation · HECKTOR · Deep learning

# Our Step By Step Process 

To replicate results, please go through the below process in order.
![alt text](https://github.com/xmuyzz/HECKTOR2022/blob/master/dataflowdiagram.png?raw=true)

### 1. Install Dependencies

```pip install -r requirements.txt```

### 2. Preprocess the Datset

Run ```python ...py``` to register, resample, and crop the data into a usable `datasets` folder.

### 3. Train the models
For SwinUNet run:

```python swin_unet.py```

For nnUNet and nnMNet: 

Follow the detailed instructions to run the code in the provided notebooks

### 4. Evaluate Performance

Evalate the aggregated metrics using the scripts in `evaluation/diceaggregated.py`

### 5. Post Processing

Run `...py` to reverse the preprocessing steps and align the masks to the orientation and dimensions of the original images

# Models We Implemented for HECKTOR
 - 2D nnUNet
 - <b> 3D nnUNet </b>  (used as official submission)
 - MNet 
 - <b> Swin Transformer </b> (used as official submission) 
 - <b> nnMNet </b> (used as official submission) 
 - 3D UNet 
 - AttUNet
 - ResUNet

nnUNet citation: https://github.com/MIC-DKFZ/nnUNet <br>
MNet citation: https://github.com/zfdong-code/mnet <br>
Swin Transformer: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR

# HECKTOR2022 Challenge Information
https://hecktor.grand-challenge.org/Timeline/ 

May 24th, 2022: Registration to the challenge opens
June 1st, 2022 June 7th, 2022: Release of the training cases.
August 1st, 2022: Release of the testing cases.
August 26th to September 2nd, 2022: Challenge submissions.
September 2nd, 2022: Paper abstract submisison deadline (intent to submit a paper).
September 8th, 2022: Full paper submission deadline.
September 8th to October 28th, 2022: Paper review phase.
September 19-22, 2022: MICCAI event and release of challenge ranking. 

In 2020, we organized the HECKTOR challenge to offer an opportunity for participants working on 3D segmentation algorithms to develop automatic bi-modal approaches for the segmentation of H&N primary tumors (GTVp) in PET/CT scans, focusing on oropharyngeal cancers [Oreiller et al. 2022]. In the 2021 edition, the scope of the challenge was expanded by proposing the segmentation task on a larger population (adding patients from a new clinical center) as well as adding a second task (split into two subtasks: with or without reference contours provided), namely the prediction of Progression-Free Survival (PFS) [Andrearczyk et al. 2022].



For the 2022 third edition of HECKTOR, we propose expanding the scope of the challenge even further by:

Adding the task of H&N nodal Gross Tumor Volumes (GTVn) segmentation (which also carries information on the outcome via the presence and amount of metastatic lesions);
Adding more data with an approximative total of 882 cases (versus 325 in 2021), including three new centers. The test data of 2021 is moved to the training set of 2022, whereas new data from new centers are split between training and test sets;
Not providing bounding boxes. Only the entire PET/CT images are provided to the challengers for fully automatic algorithms that can perform predictions from entire images;
Not providing test ground truth tumor delineations for outcome prediction, as we learned from 2021’s results that these delineations were not necessarily needed to achieve best prediction performance.'



Task 1: Primary tumor (GTVp) and lymph nodes (GTVn) segmentation in PET/CT images.
Task 2: Recurrence-Free Survival (RFS) prediction relying on PET/CT images and/or available clinical information.
see details of tasks at https://hecktor.grand-challenge.org/Evaluation/


