import os

import cv2
from deepface import DeepFace


def select_face(image_path):
    # original image aspect ratio
    origin_image = cv2.imread(image_path)
    print(image_path)
    # origin_coord = origin_image.shape
    # origin_aspect_ratio = origin_coord[1] / origin_coord[0]

    # find face
    result = DeepFace.extract_faces(
        img_path=image_path, detector_backend="retinaface", enforce_detection=False
    )

    # face aspect ratio
    face_coord = result[0]["facial_area"]
    face_x, face_y, face_w, face_h = (
        face_coord["x"],
        face_coord["y"],
        face_coord["w"],
        face_coord["h"],
    )
    # face_aspect_ratio = face_w / face_h

    # corrections for face
    # Increase size by 50%
    scale = 1.5
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)

    # Find center of original crop
    center_x = face_x + face_w // 2
    center_y = face_y + face_h // 2

    # Compute new top-left corner
    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)

    # Ensure it does not exceed original image dimensions
    new_x2 = min(origin_image.shape[1], new_x + new_w)
    new_y2 = min(origin_image.shape[0], new_y + new_h)

    # Perform cropping
    cropped_image = origin_image[new_y:new_y2, new_x:new_x2]

    return cropped_image


for filename in os.listdir("originalimages"):
    if filename.lower().endswith((".png", ".jpg")):  # Filter image files
        img_path = os.path.join("originalimages", filename)

    cv2.imwrite(f"croppedimages/cropped_{filename}", select_face(img_path))
    print("successful crop")
