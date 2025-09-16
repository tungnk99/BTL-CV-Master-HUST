import cv2

def draw_index_contours(img, contours):
    contours_with_bbox = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        contours_with_bbox.append((cnt, x, y, w, h))

    # Sắp xếp: trước tiên theo y (từ trên xuống), sau đó theo x (trái qua phải)
    contours_sorted = sorted(contours_with_bbox, key=lambda b: (b[2], b[1]))

    # Vẽ và đánh số
    for idx, (cnt, x, y, w, h) in enumerate(contours_sorted, start=1):
        # Vẽ contour
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

        # Tâm chữ
        cx, cy = x + w // 2, y + h // 2

        # Đặt số
        cv2.putText(img, str(idx), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img