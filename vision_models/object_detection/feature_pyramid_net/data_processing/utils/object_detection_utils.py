#
# MIT License
#
# Copyright (c) 2019 Sagar Vinodababu
# Portions Copyright (C) 2022-2024 Maxim Integrated Products, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# GitHub repo for the following helper methods
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
#
""" Utility functions for Object Detection Tasks """

import torch
import torch.nn.functional as F
from math import sqrt

def detect_objects(predicted_locs, predicted_scores, num_classes, prior_boxes, min_score,
                    max_overlap, top_k, return_kpts=False):
    """
    Decipher the locations and class scores to detect objects.
    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum
    threshold.
    :param predicted_locs: predicted locations/boxes w.r.t the prior boxes, a tensor of
    dimensions
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of
    dimensions
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score
    is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the
    top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    n_priors = prior_boxes.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = []
    all_images_kpts = []
    all_images_labels = []
    all_images_scores = []

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], prior_boxes))
        box_kpts = predicted_locs[i, :, 4:]

        # Lists to store boxes and scores for this image
        image_boxes = []
        image_kpts = []
        image_labels = []
        image_scores = []

        # Check for each class
        for c in range(1, num_classes):
            # Keep only predicted boxes and scores where scores for this class are above the
            # minimum score
            class_scores = predicted_scores[i][:, c]
            score_above_min_score = class_scores > min_score
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
            class_box_kpts = box_kpts[score_above_min_score]  # (n_qualified, 8)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
            # (n_qualified), (n_min_score)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)
            class_box_kpts = class_box_kpts[sort_ind]  # (n_min_score, 8)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
            # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)

            # A torch.bool tensor to keep track of which predicted boxes to suppress
            # True implies suppress, False implies don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.bool)
            # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box]:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum
                # overlap
                # Find such boxes and update suppress indices
                suppress = torch.logical_or(suppress, overlap[box] > max_overlap)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = False

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[~suppress])
            image_kpts.append(class_box_kpts[~suppress])
            image_labels.append(
                torch.LongTensor((~suppress).sum().item() * [c]))
            image_scores.append(class_scores[~suppress])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]))
            image_kpts.append(torch.FloatTensor([[0., 0., 1., 0., 0., 1., 1., 1.]])
                                )
            image_labels.append(torch.LongTensor([0]))
            image_scores.append(torch.FloatTensor([0.]))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_kpts = torch.cat(image_kpts, dim=0)  # (n_objects, 8)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_kpts = image_kpts[sort_ind][:top_k]  # (top_k, 8)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_kpts.append(image_kpts)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    if return_kpts:
        # lists of length batch_size
        return all_images_boxes, all_images_kpts, all_images_labels, all_images_scores

    # lists of length batch_size
    return all_images_boxes, all_images_labels, all_images_scores

def create_prior_boxes():
        """
        Create the prior (default) boxes
        :return: prior boxes in center-size coordinates
        """

        fmap_dims = {'f0': (32, 40),
                     'f1': (16, 20),
                     'f2': (8, 10),
                     'f3': (4, 5)}

        fmap_dim_scales = {'f0': 0.1,
                           'f1': 0.2,
                           'f2': 0.4,
                           'f3': 0.8}

        fmaps = list(fmap_dims.keys())

        obj_scales = {'s0': 2 ** 0,
                      's1': 2 ** (1.5 / 3.0),
                      }

        aspect_ratios = {'ar0': 0.5,
                         'ar1': 1,
                         'ar2': 2
                         }

        prior_boxes = []

        for fmap in fmaps:
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    cx = (j + 0.5) / fmap_dims[fmap][1]
                    cy = (i + 0.5) / fmap_dims[fmap][0]

                    for ratio in aspect_ratios.values():
                        for obj_scale in obj_scales.values():

                            prior_boxes.append([cx, cy,
                                                (obj_scale*fmap_dim_scales[fmap]) * sqrt(ratio),
                                                (obj_scale*fmap_dim_scales[fmap]) / sqrt(ratio)])

        prior_boxes = torch.tensor(prior_boxes, dtype=torch.float)
        prior_boxes.clamp_(0, 1)  # (num_priors, 4)

        return prior_boxes

def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function
    (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """
    images = []
    boxes_and_labels = []
    for b in batch:
        images.append(b[0])
        boxes_and_labels.append(b[1])
    images = torch.stack(images, dim=0)
    return images, boxes_and_labels


def check_target_exists(target_list):
    """
    Checks whether any object exists in given target list
    Object detection data loaders return target as
        target[0]: boxes list
        target[1]: labels list
    For images without any objects, these lists are both empty
    target_list is list of targets e.g. targets in given batch
    """
    for target in target_list:
        if target[0].numel() > 0:
            return True
    return False


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size
    coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates
    (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes
    (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the
    size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor
    of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """
    eps = 1e-7
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo,
    # completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                      torch.log((cxcy[:, 2:] / priors_cxcy[:, 2:]) + eps) * 5], 1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form
    mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size
    (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:4] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:4] / 5) * priors_cxcy[:, 2:4]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary
    coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2,
    a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in
    boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in
    set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)
