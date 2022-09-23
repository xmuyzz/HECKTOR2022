import numpy as np
import math
#
# import surface_dice_implementation
# from calculate_bbox_metrics import calculate_bbox_metrics


def precision(gt, pr):
    TP = np.logical_and(gt, pr).sum()
    FP = pr[(pr==1) & (gt==0)].sum()
    deno = TP+FP
    if deno == 0:
        return np.NaN
    return TP/deno


def recall(gt, pr):
    TP = np.logical_and(gt, pr).sum()
    FN = gt[(gt==1) & (pr==0)].sum()
    deno = TP+FN
    if deno == 0:
        return np.NaN
    return TP/deno


def jaccard(gt, pr):
    TP = np.logical_and(gt, pr).sum()
    FP = pr[(pr==1) & (gt==0)].sum()
    FN = gt[(gt==1) & (pr==0)].sum()
    deno = TP+FP+FN
    if deno == 0:
        return np.NaN
    return TP/deno


def dice(gt, pr):
    TP = np.logical_and(gt, pr).sum()
    FP = pr[(pr==1) & (gt==0)].sum()
    FN = gt[(gt==1) & (pr==0)].sum()
    deno = (2*TP)+FP+FN
    if deno == 0:
        return np.NaN
    return (2*TP)/deno


# original implementation in java
# double score = 0;
# if (tp > 0) {
#     double t = tp + fn;
#     double fn2 = fn * t / tp;
#     double e = fn2 + fp;
#     double a = Math.pow(36 * Math.PI * t * t, (double)1 / 3);
#     double exp1 = e / t;
#     double exp2 = e / (10 * a);
#     score = Math.exp(-(exp1 + exp2)/2);
# 
# new implementation based on java code
# Use this one


def segmentation_score(gt, pr, spacing):
    """
        gt: numpy array of ground truth
        pr: numpy array of prediction
        spacing: list of z,y,x spacing in mm (from util func)
    """
    voxel_volume = np.prod(spacing)
    TP = np.logical_and(gt, pr).sum() * voxel_volume
    FN = gt[(gt==1) & (pr==0)].sum() * voxel_volume
    FP = pr[(pr==1) & (gt==0)].sum() * voxel_volume
    if TP > 0:
        T = TP + FN
        FN2 = FN * T / TP
        E = FN2 + FP
        # E correct
        A = (36 * math.pi * T * T) ** (1/3)
        EXP1 = E / T
        EXP2 = E / (10 * A)
        return math.exp( -(EXP1 + EXP2)/2)
    else:
        return np.NaN


# old implementation based on supplement equation
# not used for now, but identical results to abv
def segmentation_score_2(gt, pr, spacing):
    """
        gt: numpy array of ground truth
        pr: numpy array of prediction
        spacing: list of z,y,x spacing in mm (from util func)
    """
    voxel_volume = np.prod(spacing)
    TP = np.logical_and(gt, pr).sum() * voxel_volume
    FN = gt[(gt==1) & (pr==0)].sum() * voxel_volume
    FP = pr[(pr==1) & (gt==0)].sum() * voxel_volume
    if TP > 0:
        V = gt[gt==1].sum() * voxel_volume # assuming volume of gt
        E = (V * FN/TP) + FP # non-negative error volume
        # E correct
        V0 = ((4*math.pi)/3) * (30**3) # scale parameter
        return math.exp (-1 * (E/(2*V)) * (1 + (V/V0)**(1/3)))
    else:
        return np.NaN

# def bbox_distance(gt, pr, spacing):
#     return calculate_bbox_metrics(gt, pr, spacing)

# def surface_dice(gt, pr, spacing, hausdorff_percent, overlap_tolerance, surface_dice_tolerance):
#     """
#         gt: numpy array of ground truth
#         pr: numpy array of prediction
#         spacing: list of z,y,x spacing in mm (from util func)
#         hausdorff_percent: percentile at which to calculate the Hausdorff distance (which is usually calculated as the maximum distance)
#         overlap_tolerance: float (mm) of what is considered as "overlap"
#         surface_dice_tolerance: float (mm) same as overlap tolerance but used for surface dice calculation
#     """
#     gt = gt.astype(np.bool)
#     pr = pr.astype(np.bool)
#     #
#     surface_distances = surface_dice_implementation.compute_surface_distances( gt, pr, spacing)
#     #
#     average_surface_distance_gt_to_pr,average_surface_distance_pr_to_gt = surface_dice_implementation.compute_average_surface_distance( surface_distances)
#     #
#     robust_hausdorff = surface_dice_implementation.compute_robust_hausdorff(
#     surface_distances, hausdorff_percent)
#     #
#     overlap_fraction_gt_with_pr, overlap_fraction_pr_with_gt =  surface_dice_implementation.compute_surface_overlap_at_tolerance(
#     surface_distances, overlap_tolerance)
#     #
#     surface_dice = surface_dice_implementation.compute_surface_dice_at_tolerance(
#     surface_distances, surface_dice_tolerance)

#     return {
#         "average_surface_distance_gt_to_pr": average_surface_distance_gt_to_pr,
#         "average_surface_distance_pr_to_gt": average_surface_distance_pr_to_gt,
#         "robust_hausdorff": robust_hausdorff,
#         "overlap_fraction_gt_with_pr": overlap_fraction_gt_with_pr,
#         "overlap_fraction_pr_with_gt": overlap_fraction_pr_with_gt,
#         "surface_dice": surface_dice
#     }

# backup dice function
# https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.
    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum






