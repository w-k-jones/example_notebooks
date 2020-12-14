import numpy as np
from scipy import ndimage as ndi

def apply_func_to_labels(labels, field, func):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array([func(field.ravel()[args[bins[i]:bins[i+1]]])
                     if bins[i+1]>bins[i] else None for i in range(bins.size-1)])

def apply_weighted_func_to_labels(labels, field, weights, func):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array([func(field.ravel()[args[bins[i]:bins[i+1]]], weights.ravel()[args[bins[i]:bins[i+1]]])
                     if bins[i+1]>bins[i] else None for i in range(bins.size-1)])

def flat_label(mask, structure=ndi.generate_binary_structure(3,1)):
    label_struct = structure.copy()
    label_struct[0] = 0
    label_struct[-1] = 0

    return ndi.label(mask, structure=label_struct)[0]

def get_step_labels_for_label(labels, structure=ndi.generate_binary_structure(3,1)):
    step_labels = flat_label(labels!=0)
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return [np.unique(step_labels.ravel()[args[bins[i]:bins[i+1]]])
            if bins[i+1]>bins[i] else None for i in range(bins.size-1)]

def filter_labels_by_length(labels, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array([o[0].stop-o[0].start for o in ndi.find_objects(labels)])
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            if object_lengths[i]<min_length:
                labels.ravel()[args[bins[i]:bins[i+1]]] = 0
            else:
                labels.ravel()[args[bins[i]:bins[i+1]]] = counter
                counter += 1
    return labels

def filter_labels_by_length_and_mask(labels, mask, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array([o[0].stop-o[0].start for o in ndi.find_objects(labels)])
    counter = 1
    for i in range(bins.size-1):
        if bins[i+1]>bins[i]:
            if object_lengths[i]>=min_length and np.any(mask.ravel()[args[bins[i]:bins[i+1]]]):
                labels.ravel()[args[bins[i]:bins[i+1]]] = counter
                counter += 1
            else:
                labels.ravel()[args[bins[i]:bins[i+1]]] = 0
    return labels
