from deepface import DeepFace
import cv2
import os

def select_face(image_path):

    #original image aspect ratio
    origin_image = cv2.imread(image_path)
    print(image_path)
    origin_coord = origin_image.shape
    origin_aspect_ratio = origin_coord[1] / origin_coord[0]

    #find face
    result = DeepFace.extract_faces(img_path=image_path, detector_backend="retinaface", enforce_detection=False)

    #face aspect ratio
    face_coord = result[0]["facial_area"]
    face_x, face_y, face_w, face_h = face_coord["x"], face_coord["y"], face_coord["w"], face_coord["h"]
    face_aspect_ratio = face_w / face_h


    #corrections for face
    if face_aspect_ratio > origin_aspect_ratio:
        #image too wide -> increase height
        face_h_new = int(face_w / origin_aspect_ratio) #new height
        pad_vertical = (face_h_new - face_h)
        pad_horizontal = 0 #no width change
    else:
        #image too tall -> increase width
        face_w_new = int(face_h * origin_aspect_ratio)
        pad_horizontal = (face_w_new - face_w)
        pad_vertical = 0

    cropped_image = origin_image[face_y - (int(pad_vertical/2)): face_y + face_h + pad_vertical, 
                                face_x - (int(pad_horizontal/2)): face_x + face_w + pad_horizontal]

    return cropped_image




for filename in os.listdir("originalimages"):
    if filename.lower().endswith(('.png', '.jpg')):  # Filter image files
        img_path = os.path.join("originalimages", filename)
    
    cv2.imwrite(f"croppedimages/cropped_{filename}", select_face(img_path))
    print("successful crop")



