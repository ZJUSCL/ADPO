def iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area1 <= 0 or area2 <= 0:
        return 0.0

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return float(inter) / union


def majority_vote(box_lists, iou_threshold=0.5, min_votes=2):
    """
    Perform majority voting over multiple per-sample box lists.

    Args:
        box_lists: 2-D list; each inner list contains boxes in [x1, y1, x2, y2] format.
        iou_threshold: IoU threshold for grouping boxes into the same cluster.
        min_votes: minimum number of votes required to keep a cluster.

    Returns:
        List of surviving cluster centers.
    """
    if not box_lists or not any(box_lists):
        return []

    all_boxes = [box for box_list in box_lists for box in box_list]
    if not all_boxes:
        return []

    clusters = []

    for box in all_boxes:
        best_cluster_idx = -1
        best_iou = 0

        for i, cluster in enumerate(clusters):
            current_iou = iou(box, cluster['center'])
            if current_iou > best_iou and current_iou > iou_threshold:
                best_iou = current_iou
                best_cluster_idx = i

        if best_cluster_idx >= 0:
            clusters[best_cluster_idx]['boxes'].append(box)
            clusters[best_cluster_idx]['votes'] += 1
            boxes_in_cluster = clusters[best_cluster_idx]['boxes']
            n = len(boxes_in_cluster)
            clusters[best_cluster_idx]['center'] = [
                sum(b[k] for b in boxes_in_cluster) / n for k in range(4)
            ]
        else:
            clusters.append({'center': box, 'boxes': [box], 'votes': 1})

    return [c['center'] for c in clusters if c['votes'] >= min_votes]


def majority_vote_weighted(box_lists, iou_threshold=0.5, min_votes=2, confidence_scores=None):
    """
    Weighted majority voting over multiple per-sample box lists.

    Args:
        box_lists: 2-D list; each inner list contains boxes in [x1, y1, x2, y2] format.
        iou_threshold: IoU threshold for grouping boxes into the same cluster.
        min_votes: minimum number of votes required to keep a cluster.
        confidence_scores: optional per-list confidence weights.

    Returns:
        List of surviving cluster centers with appended normalized confidence score.
    """
    if not box_lists or not any(box_lists):
        return []

    all_boxes_with_weights = [
        (box, confidence_scores[i] if confidence_scores else 1.0)
        for i, box_list in enumerate(box_lists)
        for box in box_list
    ]
    if not all_boxes_with_weights:
        return []

    clusters = []

    for box, weight in all_boxes_with_weights:
        best_cluster_idx = -1
        best_iou = 0

        for i, cluster in enumerate(clusters):
            current_iou = iou(box, cluster['center'])
            if current_iou > best_iou and current_iou > iou_threshold:
                best_iou = current_iou
                best_cluster_idx = i

        if best_cluster_idx >= 0:
            cluster = clusters[best_cluster_idx]
            cluster['boxes'].append(box)
            cluster['weights'].append(weight)
            cluster['total_weight'] += weight
            cluster['votes'] += 1
            total_weight = cluster['total_weight']
            cluster['center'] = [
                sum(b[k] * w for b, w in zip(cluster['boxes'], cluster['weights'])) / total_weight
                for k in range(4)
            ]
        else:
            clusters.append({
                'center': box,
                'boxes': [box],
                'weights': [weight],
                'total_weight': weight,
                'votes': 1,
            })

    return [
        c['center'] + [c['total_weight'] / len(box_lists)]
        for c in clusters
        if c['votes'] >= min_votes
    ]


if __name__ == "__main__":
    model1_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    model2_boxes = [[105, 95, 205, 195], [295, 305, 395, 405]]
    model3_boxes = [[95, 105, 195, 205], [500, 500, 600, 600]]
    box_lists = [model1_boxes, model2_boxes, model3_boxes]

    voted_boxes = majority_vote(box_lists, iou_threshold=0.5, min_votes=2)
    print("Voted boxes:", voted_boxes)

    confidence_scores = [0.9, 0.8, 0.7]
    weighted_voted_boxes = majority_vote_weighted(
        box_lists, iou_threshold=0.5, min_votes=2, confidence_scores=confidence_scores
    )
    print("Weighted voted boxes:", weighted_voted_boxes)
