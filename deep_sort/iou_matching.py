# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # print("input to IOU: bbox, candidates", bbox, candidates)

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        assert False
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        assert False
        detection_indices = np.arange(len(detections))


    # print("iou cost idxs", len(track_indices), len(detection_indices))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        # print("index", row, "bbox", bbox, "candidates", candidates)
        scores = iou(bbox, candidates)
        # print("scores", scores)
        cost_matrix[row, :] = 1. - scores
    # print("iou cost:", cost_matrix)
    return cost_matrix

if __name__ == "__main__":

    box = [  403 ,231 ,24 ,72  ]
    candidates = \
    [  [411 ,197, 46 ,118 ],
    [856, 229, 33 ,77 ],
    [757 ,238, 22, 60 ]]

    # print(iou(np.array(box), np.array(candidates)))

    # index 1
    # box 855.395 225.165 26.3955 82.8351 
    #  411 197 46 118 
    #  856 229 33 77 
    #  757 238 22 60 
    # index 2
    # box 750.838 236.835 21.4897 61.3299 
    #  411 197 46 118 
    #  856 229 33 77 
    #  757 238 22 60 

        # iou()