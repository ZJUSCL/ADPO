import math

def ciou(box1, box2):
    """
    Compute Complete IoU (CIoU) between two bounding boxes.

    Args:
        box1, box2: bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        CIoU value in [-1, 1].
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    iou = inter_area / union_area

    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    center_distance_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2

    enclose_x1 = min(box1[0], box2[0])
    enclose_y1 = min(box1[1], box2[1])
    enclose_x2 = max(box1[2], box2[2])
    enclose_y2 = max(box1[3], box2[3])
    enclose_diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    if h1 == 0 or h2 == 0:
        v = 0
        alpha = 0
    else:
        v = (4 / (math.pi ** 2)) * ((math.atan(w1 / h1) - math.atan(w2 / h2)) ** 2)
        alpha = v / (1 - iou + v + 1e-8)

    if enclose_diagonal_sq == 0:
        ciou_value = iou
    else:
        ciou_value = iou - center_distance_sq / enclose_diagonal_sq - alpha * v

    return ciou_value


if __name__ == "__main__":
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    print(f"CIoU = {ciou(box1, box2):.4f}")

    box3 = [0, 0, 10, 10]
    print(f"CIoU (full overlap) = {ciou(box3, box3):.4f}")

    box4 = [0, 0, 5, 5]
    box5 = [10, 10, 15, 15]
    print(f"CIoU (no overlap) = {ciou(box4, box5):.4f}")
