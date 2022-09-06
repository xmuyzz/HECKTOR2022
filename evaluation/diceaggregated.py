import nibabel as nib
import numpy as np
import os
import cv2
from tqdm import tqdm
import metrics
import json
import pandas as pd
import glob


def get_scores_onelabel(pred_label_dir, gt_label_dir, save_format='csv'):
    
    print("\nConverting all predictions to one label...")
    all_scores = {
        'scan': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'jaccard': [],
        'segmentation': []}
    emptys = []
    for ids, scan in enumerate(tqdm(sorted(os.listdir(pred_label_dir)))):        
        if ".nii.gz" not in scan:
            continue
        print("Looking at scan:", scan)
        scan_pred = nib.load(os.path.join(pred_label_dir, scan)).get_fdata()
        gt = nib.load(os.path.join(gt_label_dir, scan)).get_fdata()
        pred = np.where(scan_pred != 0.0, 1, 0)
        gt_seg = np.where(gt != 0.0, 1, 0)
        assert pred.shape == gt_seg.shape
        # get metrics
        if np.all(gt_seg==0):
            emptys.append(scan)
            print('empty mask:', scan)
        else:
            dice_score = metrics.dice(pred, gt_seg)
            precision_score = metrics.precision(pred, gt_seg)
            recall_score = metrics.recall(pred, gt_seg)
            jaccard_score = metrics.jaccard(pred, gt_seg)
            segm_score = metrics.segmentation_score(pred, gt_seg, [1, 1, 1])
            # print(dice_score,precision_score,recall_score,jaccard_score,segm_score)
            # if np.count_nonzero(gt_seg) > 100:  
            # append scores
            all_scores['scan'].append(scan)
            all_scores['dice'].append(dice_score)
            all_scores['precision'].append(precision_score)
            all_scores['recall'].append(recall_score)
            all_scores['jaccard'].append(jaccard_score)
            all_scores['segmentation'].append(segm_score)
    print('empty mask:', len(emptys), emptys)
    print(len([x for x in all_scores['dice'] if np.isnan(x) == False]))
    print(all_scores['dice'])
    dice_mean = np.mean([x for x in all_scores['dice'] if np.isnan(x) == False])
    precision_mean = np.mean([x for x in all_scores['precision'] if np.isnan(x) == False])
    recall_mean = np.mean([x for x in all_scores['recall'] if np.isnan(x) == False])
    jaccard_mean = np.mean([x for x in all_scores['jaccard'] if np.isnan(x) == False])
    seg_score_mean = np.mean([x for x in all_scores['segmentation'] if np.isnan(x) == False])
    print("Mean DICE for all classes:", round(dice_mean, 3))
    print("Mean Precision for all classes:", round(precision_mean, 3))
    print("Mean Recall for all classes:", round(recall_mean, 3))
    print("Mean Jaccard for all classes:", round(jaccard_mean, 3))
    print("Mean Segmentation Score for all classes:", round(seg_score_mean, 3))
    if save_format == 'json':
        fn = pred_label_dir + '/all_scores.json'
        with open(fn, 'w') as fp:
            json.dump(all_scores, fp)
        print('json file has been saved!')
    elif save_format == 'csv':
        df = pd.DataFrame.from_dict(all_scores)
        df = df.round(3)
        fn = pred_label_dir + '/all_scores.csv'
        df.to_csv(fn, index=False)
        print('csv has been saved!')


def get_scores(pred_label_dir, gt_label_dir):
    
    print("\nUsing all available labels...")
    all_scores = {
        'dice':[],
        'precision':[],
        'recall':[],
        'jaccard':[],
        'segmentation':[]}
    indiv_scores = {
        'dice':{0:[],1:[],2:[],3:[],},
        'precision':{0:[],1:[],2:[],3:[],},
        'recall':{0:[],1:[],2:[],3:[],},
        'jaccard':{0:[],1:[],2:[],3:[],},
        'segmentation':{0:[],1:[],2:[],3:[],}}
    # indiv_dices = {0:[],1:[],2:[],3:[],} 
    for ids, scan in enumerate(tqdm(os.listdir(pred_label_dir))):    
        if ".nii.gz" not in scan:
            continue
        print("Looking at scan:",scan)
        scan_pred = nib.load(os.path.join(pred_label_dir, scan)).get_fdata()
        gt = nib.load(os.path.join(gt_label_dir, scan)).get_fdata()
        for i in range(4):
            pred = np.where(scan_pred==i, 1, 0)
            gt_seg = np.where(gt==i, 1, 0)
            dice_score = metrics.dice(pred, gt_seg)
            precision_score = metrics.precision(pred, gt_seg)
            recall_score = metrics.recall(pred, gt_seg)
            jaccard_score = metrics.jaccard(pred, gt_seg)
            segm_score = metrics.segmentation_score(pred, gt_seg, [1, 1, 1])
            # if np.count_nonzero(gt_seg) > 100:
                # print("\tDice for label",i,":",dice(pred,gt_seg))
            indiv_scores['dice'][i].append(dice_score)
            indiv_scores['precision'][i].append(precision_score)
            indiv_scores['recall'][i].append(recall_score)
            indiv_scores['jaccard'][i].append(jaccard_score)
            indiv_scores['segmentation'][i].append(segm_score)
            if i != 0:
                all_scores['dice'].append(dice_score)
                all_scores['precision'].append(precision_score)
                all_scores['recall'].append(recall_score)
                all_scores['jaccard'].append(jaccard_score)
                all_scores['segmentation'].append(segm_score)
        # print(scan_pred.shape,np.where(scan_pred==1.0, True, False).shape)
        # print(gt.shape,gt[gt==1.0].shape)
    dice_mean = np.mean([x for x in all_scores['dice'] if np.isnan(x) == False])
    precision_mean = np.mean([x for x in all_scores['precision'] if np.isnan(x) == False])
    recall_mean = np.mean([x for x in all_scores['recall'] if np.isnan(x) == False])
    jaccard_mean = np.mean([x for x in all_scores['jaccard'] if np.isnan(x) == False])
    seg_score_mean = np.mean([x for x in all_scores['segmentation'] if np.isnan(x) == False])
    dice0 = np.mean([x for x in indiv_scores['dice'][0] if np.isnan(x) == False])
    dice1 = np.mean([x for x in indiv_scores['dice'][1] if np.isnan(x) == False])
    dice2 = np.mean([x for x in indiv_scores['dice'][2] if np.isnan(x) == False])
    print("Mean DICE for all classes:", round(dice_mean, 3))
    print("Mean DICE for background:", round(dice0, 3))
    print("Mean DICE for label 1:", round(dice1, 3))
    print("Mean DICE for label 2:", round(dice2, 3))
    print("Mean Precision for all classes:", round(precision_mean, 3))
    print("Mean Recall for all classes:", round(recall_mean, 3))
    print("Mean Jaccard for all classes:", round(jaccard_mean, 3))
    print("Mean Segmentation Score for all classes:", round(seg_score_mean, 3))


def get_aggregated_dice(pred_label_dir, gt_label_dir):
    
    inters = []
    sums = []
    unions = []
    for pred_dir in tqdm(sorted(glob.glob(pred_label_dir + '/*.nii.gz'))):
        fn = pred_dir.split('/')[-1]
        print('Looking at case:', fn)
        pred = nib.load(pred_dir).get_fdata()
        gt = nib.load(os.path.join(gt_label_dir, fn)).get_fdata()
        pred = np.where(scan_pred != 0.0, 1, 0)
        gt_seg = np.where(gt != 0.0, 1, 0)
        assert pred.shape == gt.shape, print('different shape')
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        volume_sum = gt.sum() + pred.sum()
        volume_intersect = (gt & pred).sum()
        volume_union = volume_sum - volume_intersect
        inters.append(volume_intersect)
        sums.append(volume_sum)
        unions.append(volume_union)
    #print('sums:', sums)
    #print('unions:', unions)
    #print('intersections:', inters)
    dsc_agg = round(2*sum(inters)/sum(sums), 3)
    jaccard_agg = round(sum(inters)/sum(unions), 3)
    print('aggregated dice score:', dsc_agg)
    print('aggregated jaccard score:', jaccard_agg)

    return dsc_agg, jaccard_agg


def get_aggregated_dice2(pred_label_dir, gt_label_dir):

    interss = []
    sumss = []
    unionss = []
    for i in [1, 2]:
        inters = []
        sums = []
        unions = []
        for pred_dir in tqdm(sorted(glob.glob(pred_label_dir + '/*.nii.gz'))):
            print(pred_dir)
            fn = pred_dir.split('/')[-1]
            print('Looking at case:', fn)
            pred = nib.load(pred_dir).get_fdata()
            gt = nib.load(os.path.join(gt_label_dir, fn)).get_fdata()
            pred = np.where(pred == i, 1, 0)
            gt = np.where(gt == i, 1, 0)
            assert pred.shape == gt.shape, print('different shape')
            pred = pred.astype(bool)
            gt = gt.astype(bool)
            volume_sum = gt.sum() + pred.sum()
            volume_intersect = (gt & pred).sum()
            volume_union = volume_sum - volume_intersect
            inters.append(volume_intersect)
            sums.append(volume_sum)
            unions.append(volume_union)
        interss.append(inters)
        sumss.append(sums)
        unionss.append(unions)
    dsc_agg1 = round(2*sum(interss[0])/sum(sumss[0]), 3)
    jaccard_agg1 = round(sum(interss[0])/sum(unionss[0]), 3)
    print('label 1 aggregated ice score:', dsc_agg1)
    print('label 1 aggregated jaccard score:', jaccard_agg1)
    dsc_agg2 = round(2*sum(interss[1])/sum(sumss[1]), 3)
    jaccard_agg2 = round(sum(interss[1])/sum(unionss[1]), 3)
    print('label 2 aggregated dice score:', dsc_agg2)
    print('label 2 aggregated jaccard score:', jaccard_agg2)


if __name__ == "__main__":
    
    #data_dir = '/mnt/aertslab/USERS/Zezhong/hecktor2022/DATA2'
    #data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet'
    #gt_label_dir = data_dir + '/nnUNet_raw_data_base/nnUNet_raw_data/Task504_primary_CT/labelsTs'
    #pred_label_dir = data_dir + '/all_results/primary_CT'
    #gt_label_dir = data_dir + '/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Hecktor/labelsTs'
    #pred_label_dir = data_dir + '/results_test'
    gt_label_dir = "/content/drive/MyDrive/HECKTOR2022/DATA/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Hecktor/labelsTs"
    pred_label_dir = "/content/drive/MyDrive/HECKTOR2022/DATA/RESULTS_FOLDER_MNETNNUNET/nnUNet/3d_fullres/Task501_Hecktor/myTrainer__nnUNetPlansv2.1/testall"
    #get_scores(pred_label_dir, gt_label_dir)
    #get_scores_onelabel(pred_label_dir, gt_label_dir)

    get_aggregated_dice2(pred_label_dir, gt_label_dir)
    get_scores(pred_label_dir, gt_label_dir)

    #get_aggregated_dice(pred_label_dir, gt_label_dir)
