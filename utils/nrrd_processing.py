### Initiate this file after nrrds have been generated for labels and images in run_bk file
## 
## Benjamin Kann
### Order of operations: combine label nrrds, interpolate image, interpolate combined label, crop roi

import sys
import os
import pydicom
import matplotlib
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
#sys.path.append('/data-utils')

GPU = True

if GPU==True:
    sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')
    path_input = "/media/bhkann/HN_RES/HN_PETSEG"
    path_deeplearning = "/home/bhkann/deeplearning/input/" 
else:
    sys.path.append('/Users/BHKann/git-code/hn-petct-net/data-utils')
    path_input = "/Volumes/BK_RESEARCH/HN_PETSEG/" # path to image folders Replace with CHUS, HGJ, HRM as needed
    path_deeplearning = "/Volumes/BK_RESEARCH/HN_PETSEG/" # path to label folders

from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_roi import crop_roi, crop_top, crop_top_image_only
#from nrrd_reg import nrrd_reg_rigid
#from nrrd_reg import nrrd_reg

image_type = 'CT' # 'CT' or 'PET'
database = "MDACC" ## [CHUM, CHUS, HGJ, HMR, MDACC, PMH]
dataset = "MDACC" #"HNexports" for McGill data

if image_type == 'PET':
    ### REGISTRATION for PET/CT ###
    input_dir = "/Volumes/BK_RESEARCH/HN_PETSEG/curated/" + database + "_files/0_image_raw_" + database + "/"  # path to image files Replace with CHUS, HGJ, HRM as needed
    output_dir = '/Volumes/BK_RESEARCH/HN_PETSEG/curated/' + database + '_files/0_image_raw_' + database + '_registered'
    dataset = "HNexports"
    #nrrd_reg_rigid(dataset, input_dir, output_dir)
    
#pathtest= '/data/curated/interpolated/HNexports_HN-CHUS-004_pet_interpolated_raw_raw_xx.nrrd'

PATH0 = path_input + "/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
PATH1 = path_input + "/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders

### COMBINING MASKS ###
image_type="CT"
#data_type = "combined_masks_p" ### '_n or _p or _pn'

for i in ['_p','_n','_pn']:
    data_type = "combined_masks" + i
    for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
        mask_arr = []
        if not file.startswith('.') and '_' + image_type in file:
            patient_id = file.split('_')[1]
            modality_date = file.split('_')[2]
            print("patient ID: ", patient_id, " modality: ", modality_date)
            path_image = os.path.join(PATH0,file)            
            print("image path: ",path_image)
            # 5. combine single mask nrrds into a single nrrd
            #dataset = "HNexports"
            ## For PMH set only, combine all nodes together to GTVn 
            if database == 'PMH' and (data_type == 'combined_masks_n' or data_type == 'combined_masks_pn'):
                node_list = ['LRetro','Lretro','L1','L1A','L1B','L1B2','L1B2A','L12','L2A','L2A1B','L2A5','L2A3','L2A4','L2AB3','L2B3','L2B4','L2B5','L2B','L2','L3','L4','L5','L5A','L5B','L23','L24','L25','L234','R235','L2345','L34','L45','L6','L7',
                            'RRetro','Rretro','R1','R1A','R1B','R1B2','R1B2A','R12','R2A','R2A1B','R2A3','R2A4','R2A5','R2AB3','R2B3','R2B4','R2B5','R2B','R2','R3','R4','R5','R5A','R5B','R23','R24','R25','R234','R235','R2345','R34','R45','R6','R7']
                node_list = [n + '.nrrd' for n in node_list]
                node_list2 = [n + 'GTV.nrrd' for n in node_list]
                node_list3 = ['I' + n + '.nrrd' for n in node_list] + ['II' + n + '.nrrd' for n in node_list] + ['III' + n + '.nrrd' for n in node_list] + ['IV' + n + '.nrrd' for n in node_list] + ['V' + n + '.nrrd' for n in node_list] + ['VI' + n + '.nrrd' for n in node_list] + ['VII' + n + '.nrrd' for n in node_list]
                try:
                    for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): 
                        if struct in node_list or struct.endswith(tuple(node_list)) or struct in node_list2 or struct in node_list3:
                            if 'ext' not in struct and 'EXT' not in struct and 'ref' not in struct and 'REF' not in struct:
                                mask_arr.append(PATH1 + database + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                        if data_type == 'combined_masks_pn' and (struct=='GTV.nrrd' or struct=='HTV.nrrd'):
                            mask_arr.append(PATH1 + database + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                    path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                    binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                    return_type = "sitk_object"
                    output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                    print(mask_arr)
                    combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")        
            elif database == 'PMH' and data_type == 'combined_masks_p':
                try:
                    for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        if struct == "GTV.nrrd" or struct == "HTV.nrrd":
                            mask_arr.append(PATH1 + database + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                    path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                    binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                    return_type = "sitk_object"
                    output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                    print(mask_arr)
                    combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")
            elif database == 'CHUM' and data_type == 'combined_masks_pn':
                #try:
                for folder in os.listdir(PATH1): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                    for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        if struct.startswith("GTVp") or struct.startswith("GTVn"):
                            mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                    path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                    binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                    return_type = "sitk_object"
                    output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                    print(mask_arr)
                    combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                #except: 
                #    print("combination failed.")            
            elif database == 'CHUM' and data_type == 'combined_masks_p':
                try:
                    for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        if struct.startswith("GTVp"):
                            mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                    path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                    binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                    return_type = "sitk_object"
                    output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                    print(mask_arr)
                    combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")        
            elif database == 'CHUM' and data_type == 'combined_masks_n':
                try:
                    for struct in os.listdir(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/"): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        if struct.startswith("GTVn"):
                            mask_arr.append(PATH1 + dataset + "-RTSTRUCT-SIM_" + patient_id + "/" + struct)
                    path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                    binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                    return_type = "sitk_object"
                    output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                    print(mask_arr)
                    combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")
            
            elif database == 'MDACC' and data_type == 'combined_masks_pn':
                try:
                    for folder in os.listdir(PATH1): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        mask_arr = []
                        label_id = folder.split('_')[1]
                        date_id = 'CT-' + '-'.join(folder.split('-')[2:6]) + '-'
                        pathstruct = os.path.join(PATH1,folder)
                        if label_id == patient_id and modality_date == date_id:
                            print('label_id: ', label_id, 'image_id: ', patient_id)
                            for struct in os.listdir(pathstruct):
                                if struct.startswith("GTVp") or struct.startswith("GTVn"):                        
                                    mask_arr.append(os.path.join(pathstruct,struct))
                            path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                            binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                            return_type = "sitk_object"
                            output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                            print(mask_arr)
                            combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")
            elif database == 'MDACC' and data_type == 'combined_masks_p':
                try:
                    for folder in os.listdir(PATH1): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        mask_arr = []
                        label_id = folder.split('_')[1]
                        date_id = 'CT-' + '-'.join(folder.split('-')[2:6]) + '-'
                        pathstruct = os.path.join(PATH1,folder)
                        if label_id == patient_id and modality_date == date_id:
                            print('label_id: ', label_id, 'image_id: ', patient_id)
                            for struct in os.listdir(pathstruct):
                                if struct.startswith("GTVp"):                        
                                    mask_arr.append(os.path.join(pathstruct,struct))
                            path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                            binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                            return_type = "sitk_object"
                            output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                            print(mask_arr)
                            combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")             
            elif database == 'MDACC' and data_type == 'combined_masks_n':
                try:
                    for folder in os.listdir(PATH1): # /Users/ben-mac/Documents/BK_RESEARCH/HN_PETSEG/curated/1_label_raw_CHUS_named/HNexports-RTSTRUCT-SIMPET_HN-CHUS-001
                        mask_arr = []
                        label_id = folder.split('_')[1]
                        date_id = 'CT-' + '-'.join(folder.split('-')[2:6]) + '-'                    
                        pathstruct = os.path.join(PATH1,folder)
                        if label_id == patient_id and modality_date == date_id:
                            print('label_id: ', label_id, 'image_id: ', patient_id)
                            for struct in os.listdir(pathstruct):
                                if struct.startswith("GTVn"):                        
                                    mask_arr.append(os.path.join(pathstruct,struct))
                            path_to_reference_image_nrrd =  path_image #"/data/curated/0_image_raw/HNexports_HN-CHUS-004_CT-SIMPET_raw_raw_raw_xx.nrrd"
                            binary = 2 #If 0, the returned array is categorical (0=background, 1=..) - single channel. If 1, the returned array is binary but in multiple channels i.e. each mask in one channel.  If 2, the returned array is binary but in a single channel i.e. masks are combined
                            return_type = "sitk_object"
                            output_dir = path_input + "/curated/" + database + "_files/combinedlabels"
                            print(mask_arr)
                            combined_mask = combine_structures(dataset, patient_id, data_type, mask_arr, path_to_reference_image_nrrd, binary, return_type, output_dir)
                except: 
                    print("combination failed.")                            


### INTERPOLATION ###            
for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
    if not file.startswith('.') and '_' + image_type in file:
        patient_id = file.split('_')[1]
        modality_date = file.split('_')[2]
        print("patient ID: ", patient_id, " modality: ", modality_date)
        path_image = os.path.join(PATH0,file)            
        print("image path: ",path_image)   
        # 6a. interpolate to a common voxel spacing - image
        data_type = "ct" #ct or pet
        # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
        path_to_nrrd = path_image
        interpolation_type = "linear" #"linear" for image, nearest neighbor for label
        spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3
        return_type = "numpy_array"
        output_dir = path_input + "/curated/" + database + "_files/interpolated"
        if database == "MDACC":
            for folder in os.listdir(PATH1):
                label_id = folder.split('_')[1]
                date_id = 'CT-' + '-'.join(folder.split('-')[2:6]) + '-'
                if label_id == patient_id and modality_date == date_id:
                    interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
        else:
            interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
        
        try:
            # 6b. interpolate to a common voxel spacing - label pn
            data_type = "label"
            # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
            path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "pn" + "_interpolated_raw_raw_xx.nrrd"
            interpolation_type = "nearest_neighbor" #"linear" for image, nearest_neighbor for label, others: bspline
            spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3
            return_type = "numpy_array"
            output_dir = path_input + "/curated/" + database + "_files/interpolated"
            interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
        except:
            print("could not interpolate pn")
        
        try:
            # 6c. interpolate to a common voxel spacing - label p
            data_type = "label_p"
            # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
            path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "p" + "_interpolated_raw_raw_xx.nrrd"
            interpolation_type = "nearest_neighbor" #"linear" for image, nearest_neighbor for label, others: bspline
            spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3
            return_type = "numpy_array"
            output_dir = path_input + "/curated/" + database + "_files/interpolated"
            interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
        except:
            print("could not interpolate p")
        
        try:
            # 6d. interpolate to a common voxel spacing - label n
            data_type = "label_n"
            # path_to_nrrd = "/data/output/dataset_124_ct_raw_raw_raw_xx.nrrd"
            path_to_nrrd = path_input + "/curated/" + database + "_files/combinedlabels/" + dataset + "_" + patient_id + "_combined_masks_" + "n" + "_interpolated_raw_raw_xx.nrrd"
            interpolation_type = "nearest_neighbor" #"linear" for image, nearest_neighbor for label, others: bspline
            spacing = (1,1,3) # x,y,z For HN SEG: 1, 1, 3
            return_type = "numpy_array"
            output_dir = path_input + "/curated/" + database + "_files/interpolated"
            interpolated_nrrd = interpolate(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, spacing, return_type, output_dir)
        except:
            print("could not interpolate n")

       

### Experimenting the alternative crops 

import sys
import os
import pydicom
import matplotlib
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import SimpleITK as sitk
#sys.path.append('/data-utils')
#sys.path.append('/Users/BHKann/git-code/hn-petct-net/data-utils')
sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')


from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_roi import crop_roi, crop_top
from nrrd_reg import nrrd_reg_rigid_ref


### Rigid Registration - followed by top crop
image_type="CT"
size_str = 'reg' #''

for database in ['MDACC']: #MDACC','PMH','CHUS','CHUM']: 
    #PATH0 = "/Volumes/BK_RESEARCH/HN_PETSEG/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
    #PATH1 = "/Volumes/BK_RESEARCH/HN_PETSEG/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders
    PATH0 = path_input + "/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
    PATH1 = path_input + "/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders    
    for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
        if not file.startswith('.') and '_' + image_type in file:
            patient_id = file.split('_')[1]
            modality = file.split('_')[2]
            print("patient ID: ", patient_id, " modality: ", modality)
            path_image = os.path.join(PATH0,file)            
            print("image path: ",path_image)   
            # 7. crop roi of defined size using a label # TRY STARTING AT SUPERIOR BORDER AND GOING DOWN 25 cm
            #Crop everything to smallest x-y use 'resize' function for this; then do z]
            #dataset = "HNexports"
            #patient_id = 
            if database == 'CHUM' or database == 'CHUS':
                dataset = 'HNexports'
            else:
                dataset = database
            path_to_image = path_input + "/curated/" + database + "_files/interpolated/" + dataset + "_" + patient_id + "_ct_interpolated_raw_raw_xx.nrrd"
            if not os.path.exists(path_input + "/curated/" + database + "_files/image" + "_" + size_str):
                os.makedirs(path_input + "/curated/" + database + "_files/image" + "_" + size_str)
            path_to_image_output = path_input + "/curated/" + database + "_files/image" + "_" + size_str
            try:
                fixed_image, moving_image, final_transform = nrrd_reg_rigid_ref(database, patient_id, path_to_image, path_to_image_output, path_input)
                print("image registered.")
            except:
                print("image registration failed")
            
            for iden in ['','_p','_n']:
                path_to_label = path_input + "/curated/" + database + "_files/interpolated/" + dataset + "_" + patient_id + "_label" + iden + "_interpolated_raw_raw_xx.nrrd"
                print("path to label: ", path_to_label)
                #plt.imshow(image_arr[15,:,:], cmap=plt.cm.gray)
                #plt.show()
                if not os.path.exists(path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str):
                    os.makedirs(path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str)                
                path_to_label_output = path_input + "/curated/" + database + "_files/label" + iden + "_" + size_str
                #try:
                #nrrd_reg_rigid_ref(database, patient_id, path_to_image, path_to_label_output, label=True, path_to_label=path_to_label)
                try:
                    moving_label = sitk.ReadImage(path_to_label, sitk.sitkFloat32)
                    moving_label_resampled = sitk.Resample(moving_label, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())
                    sitk.WriteImage(moving_label_resampled, os.path.join(path_to_label_output, patient_id + "_label_registered.nrrd"))
                    print("label registered: ", iden)
                    #transform = sitk.ReadTransform('.tfm')
                except:
                    print("label registration failed")
            
## with TOP-CROP HPC ### NEED TO RUN FOR image_crop, image_crop_p, and image_crop_n  ### WILL ONLY WORK WITH SPACING = 1,1,3
image_type="CT"
roi_size = (172,172,76) #x,y,z
size_str = '172x172x76'

for database in ['MDACC','PMH','CHUS','CHUM']: 
    PATH0 = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/0_image_raw_" + database + "/" # path to image folders Replace with CHUS, HGJ, HRM as needed
    PATH1 = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/1_label_raw_" + database + "_named/" # path to label folders
    for iden in ['','_p','_n']:
        for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
            if not file.startswith('.') and '_' + image_type in file:
                patient_id = file.split('_')[1]
                modality = file.split('_')[2]
                print("patient ID: ", patient_id, " modality: ", modality)
                path_image = os.path.join(PATH0,file)            
                print("image path: ",path_image)   
                # 7. crop roi of defined size using a label # TRY STARTING AT SUPERIOR BORDER AND GOING DOWN 25 cm
                #Crop everything to smallest x-y use 'resize' function for this; then do z]
                #dataset = "HNexports"
                #patient_id = 
                if database == 'CHUM' or database == 'CHUS':
                    dataset = 'HNexports'
                else:
                    dataset = database
                path_to_image = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_reg/" + patient_id + "_registered.nrrd"
                path_to_label = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label" + iden + "_reg/" + patient_id + "_label_registered.nrrd"
                #plt.imshow(image_arr[15,:,:], cmap=plt.cm.gray)
                #plt.show()
                try:
                    os.makedirs("/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_croptop" + "_" + size_str)
                except:
                    print("directory already exists")
                path_to_image_roi = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/image_croptop" + "_" + size_str
                try:
                    os.makedirs("/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label_croptop" + iden + "_" + size_str)                
                except:
                    print("directory already exists")
                path_to_label_roi = "/media/bhkann/HN_RES/HN_PETSEG/curated/" + database + "_files/label_croptop" + iden + "_" + size_str
                print("path_to_image: ", path_to_image)
                print("path_to_label: ", path_to_label)
                print("path_to_image_roi: ", path_to_image_roi)
                print("path_to_label_roi: ", path_to_label_roi)
                try:
                    image_obj, label_obj = crop_top(
                        dataset,
                        patient_id,
                        path_to_image,
                        path_to_label,
                        roi_size,
                        "sitk_object",
                        path_to_image_roi,
                        path_to_label_roi)
                except:
                    print("crop failed!")
                    


## FOR YALE IMAGES ONLY 
import sys
import os
import pydicom
import matplotlib
from matplotlib.path import Path
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
#sys.path.append('/data-utils')

GPU = True

if GPU==True:
    sys.path.append('/home/bhkann/git-repositories/hn-petct-net/data-utils')
    path_input = "/media/bhkann/HN_RES/HN_PETSEG"
    path_deeplearning = "/home/bhkann/deeplearning/input/" 
else:
    sys.path.append('/Users/BHKann/git-code/hn-petct-net/data-utils')
    path_input = "/Volumes/BK_RESEARCH/HN_PETSEG/" # path to image folders Replace with CHUS, HGJ, HRM as needed
    path_deeplearning = "/Volumes/BK_RESEARCH/HN_PETSEG/" # path to label folders

from dcm_to_nrrd import dcm_to_nrrd
from rtstruct_to_nrrd import rtstruct_to_nrrd
from combine_structures import combine_structures
from interpolate import interpolate
from crop_roi import crop_roi, crop_top, crop_top_image_only

image_type="ct"
roi_size = (172,172,76) #x,y,z
size_str = '172x172x76'
dataset = "Yale"
PATH0 = "/media/bhkann/HN_RES1/HN_DL/ENE_preprocess_nrrd/interpolated/" # path to image folders Replace with CHUS, HGJ, HRM as needed
for file in sorted(os.listdir(PATH0)): #LOOP Goes through nrrd raw images etc
    if not file.startswith('.') and '_' + image_type in file:
        patient_id = file.split('_')[1]
        print("patient ID: ", patient_id)
        path_image = os.path.join(PATH0,file)            
        print("image path: ",path_image)
        #plt.imshow(image_arr[15,:,:], cmap=plt.cm.gray)
        #plt.show()
        try:
            os.makedirs("/home/bhkann/deeplearning/HN_PETSEG/input/" + dataset + "_files/image_croptop" + "_" + size_str)
        except:
            print("directory already exists")
        path_to_image_cropped = "/home/bhkann/deeplearning/HN_PETSEG/input/" + dataset + "_files/image_croptop" + "_" + size_str
        print("path_to_image: ", path_image)
        print("path_to_image cropped: ", path_to_image_cropped)
        #try:
        image_obj = crop_top_image_only(
            dataset,
            patient_id,
            path_image,
            roi_size,
            "sitk_object",
            path_to_image_cropped)
        #except:
        #    print("crop failed!")
    

# crop_top_image_only(dataset, patient_id, path_to_image_nrrd, crop_shape, return_type, output_folder_image)
