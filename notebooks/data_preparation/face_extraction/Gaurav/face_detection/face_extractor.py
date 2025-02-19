# Face Detection from Video

import os
from typing import List

import cv2
import numpy as np
from imutils.video import FPS
from tqdm import tqdm

RESNET_PROTOTXT = "/workspaces/face-verifiers/models/serialized/cv2-resnet/res10_300x300_ssd_iter_140000.prototxt"
RESNET_CAFFEMODEL = "/workspaces/face-verifiers/models/serialized/cv2-resnet/res10_300x300_ssd_iter_140000.caffemodel"


class FaceDetectorResnet:
    """
    A class for detecting faces in images and videos using the ResNet-based face detection model.

    This class encapsulates the functionality to detect faces in images and videos using a
    deep learning model based on ResNet. It provides methods for processing individual
    images as well as videos, drawing bounding boxes around detected faces, and saving the
    results. The class also allows for configuring options such as scale factor and CUDA
    acceleration.

    Attributes:
        RESNET_PROTOTXT (str): The file path to the ResNet-based model's prototxt file.
        RESNET_CAFFEMODEL (str): The file path to the ResNet-based model's caffemodel file.

    Methods:
        __init__(self, scaleFactor: float = 1.0, use_cuda: bool = False):
            Initializes the FaceDetectorResnet instance with optional configuration.

        processImage(self, imgName: str) -> np.ndarray:
            Process a single image and return the image with bounding boxes around detected faces.

        processVideo_save_frames(self, videoName: str, outputFolder: str) -> None:
            Process a video, draw bounding boxes, and save frames to the specified folder.

        processVideo_save_video_with_progress(self, videoName: str, output_video_name: str) -> None:
            Process a video, save it with bounding boxes, and display a progress bar.

    Note:
        The RESNET_PROTOTXT and RESNET_CAFFEMODEL attributes should be configured to point
        to the correct model files for ResNet-based face detection.
    """

    def __init__(
        self,
        scaleFactor: float = 1.0,
        face_confidence_threshold: float = 0.99,
        use_cuda: bool = False,
    ) -> None:
        """
        Initialize the FaceDetectorResnet.

        Parameters:
        scaleFactor (float): The scale factor for scaling the bounding box.
        use_cuda (bool): Whether to use CUDA acceleration.
        """
        # Read the model
        self.faceModel = cv2.dnn.readNetFromCaffe(
            RESNET_PROTOTXT, caffeModel=RESNET_CAFFEMODEL
        )

        # Set the scale factor
        self.scaleFactor = scaleFactor

        # Set the face confidence threshold
        self.face_confidence_threshold = face_confidence_threshold

        # Enable CUDA acceleration if requested
        if use_cuda:
            print("[INFO] Using CUDA acceleration")
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, imgName: str) -> np.ndarray:
        """
        Process the given image.

        Parameters:
        imgName (str): The path to the image file.

        Returns:
        np.ndarray: The processed image.
        """

        self.img = cv2.imread(imgName)
        (self.h, self.w) = self.img.shape[:2]
        # process frame and draw bounding box around detected faces
        self.processFrame()
        return self.img

    def processFrame(self) -> None:
        """
        Process the image frame to detect faces and draw bounding boxes around them.

        This method uses the ResNet-based face detection model to identify faces in
        the input image and draws bounding boxes around detected faces.

        Returns:
        None
        """
        # Prepare the image for face detection
        blob = cv2.dnn.blobFromImage(
            self.img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        # Detect faces in the image (forward pass)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        # Iterate over detected faces and draw bounding boxes
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > self.face_confidence_threshold:
                bbox = predictions[0, 0, i, 3:7] * np.array(
                    [self.w, self.h, self.w, self.h]
                )
                (startX, startY, endX, endY) = bbox.astype("int")

                # Calculate the center, width, and height of the bounding box
                centerX = (startX + endX) // 2
                centerY = (startY + endY) // 2
                width = endX - startX
                height = endY - startY

                # Scale dimensions based on scaleFactor
                newWidth = int(width * self.scaleFactor)
                newHeight = int(height * self.scaleFactor)

                # Calculate new coordinates for the scaled bounding box
                newStartX = int(centerX - newWidth // 2)
                newStartY = int(centerY - newHeight // 2)
                newEndX = int(centerX + newWidth // 2)
                newEndY = int(centerY + newHeight // 2)

                # Draw the scaled rectangle around the detected face
                cv2.rectangle(
                    self.img, (newStartX, newStartY), (newEndX, newEndY), (0, 0, 255), 2
                )

    def processVideo_save_video(self, videoName: str, output_video_name: str) -> None:
        """
        Process the given video and save it with bounding boxes, displaying a progress bar.

        Parameters:
        videoName (str): The path to the video file.
        output_video_name (str): The name of the output video file with bounding boxes.
        """

        # Open the video file
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print("[ERROR] Opening video stream or file")
            return

        # Initialize video writer
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(
            output_video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (frame_width, frame_height),
        )

        # Initialize frame count and FPS counter
        frame_count = 0
        failed_frames = 0
        fps = FPS().start()
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framerate = int(cap.get(cv2.CAP_PROP_FPS))

        # Loop through video frames
        with tqdm(total=num_frames, desc="Processing Frames") as pbar:
            while True:
                success, self.img = cap.read()
                frame_count += 1
                pbar.update(1)

                # Handle unsuccessful frame reads
                if not success and frame_count < num_frames:
                    failed_frames += 1
                    # print(f"[ERROR] Reading frame {frame_count} for {videoName}.. skipping frame")
                    continue
                # Loop exit condition
                elif not success and frame_count == num_frames:
                    print(f"[INFO] Finished processing all frames for {videoName}")
                    break

                # Process frame
                self.h, self.w = self.img.shape[:2]
                self.processFrame()

                # Write processed frame to output video
                out.write(self.img)

                # Update FPS counter
                fps.update()

        # Release video and writer resources
        fps.stop()
        cap.release()
        out.release()

        # Display processing statistics
        video_length_minutes = num_frames / framerate / 60
        print(f"[INFO] Frames processed per second: {fps.fps():.2f}")
        print(f"[INFO] Processing time: {fps.elapsed()/60:.2f} minutes")
        print(f"[INFO] Video Length: {video_length_minutes:.2f} minutes")
        print(f"[INFO] Video framerate: {framerate:.2f} fps")
        print(
            f"[INFO] Failed frames: {failed_frames} Percentage: {failed_frames/num_frames*100:.2f}%"
        )

    def processVideo_save_frames(
        self, videoName: str, outputFolder: str, nth_frame: int = 1
    ) -> None:
        """
        Process the given video, draw bounding boxes, and save frames to the specified folder with a progress bar.

        Parameters:
        videoName (str): The path to the video file.
        outputFolder (str): The folder where frames with bounding boxes will be saved.
        nth_frame (int): The interval at which frames will be saved. Default is 1, meaning every frame will be saved.
        """

        # Open the video file
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print("[ERROR] Opening video stream or file")
            return

        # Create the output folder if it doesn't exist
        os.makedirs(outputFolder, exist_ok=True)

        # Initialize frame count and FPS counter
        frame_count = 0
        failed_frames = 0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = FPS().start()
        framerate = int(cap.get(cv2.CAP_PROP_FPS))

        # Loop through video frames
        with tqdm(total=num_frames, desc="Processing Frames") as pbar:
            while True:
                success, self.img = cap.read()
                frame_count += 1
                pbar.update(1)

                # Handle unsuccessful frame reads
                if not success and frame_count < num_frames:
                    failed_frames += 1
                    # print(f"[ERROR] Reading frame {frame_count} for {videoName}.. skipping frame")
                    continue
                # Loop exit condition
                elif not success and frame_count == num_frames:
                    print(f"[INFO] Finished processing all frames for {videoName}")
                    break

                # Process frame
                self.h, self.w = self.img.shape[:2]
                self.processFrame()

                # Save the processed frame conditionally based on nth_frame
                if frame_count % nth_frame == 0:
                    frame_filename = os.path.join(
                        outputFolder, f"frame_{frame_count:04d}.jpg"
                    )
                    cv2.imwrite(frame_filename, self.img)

                # Update FPS counter
                fps.update()

        # Release video resources
        fps.stop()
        cap.release()

        # Display processing statistics
        video_length_minutes = num_frames / framerate / 60
        print(f"[INFO] Frames processed per second: {fps.fps():.2f}")
        print(f"[INFO] Processing time: {fps.elapsed() / 60:.2f} minutes")
        print(f"[INFO] Video Length: {video_length_minutes:.2f} minutes")
        print(f"[INFO] Video framerate: {framerate:.2f} fps")
        print(
            f"[INFO] Failed frames: {failed_frames} Percentage: {failed_frames/num_frames*100:.2f}%"
        )

    def extractFace(self) -> List[np.ndarray]:
        """
        Process the image frame to detect and extract faces.

        This method uses the ResNet-based face detection model to identify faces in
        the input image. It returns the cropped face regions as a list of numpy arrays.

        Returns:
        List[np.ndarray]: List of cropped faces as numpy arrays.
        """

        cropped_faces = []  # List to store cropped faces

        # Prepare the image for face detection
        blob = cv2.dnn.blobFromImage(
            self.img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )

        # Detect faces in the image (forward pass)
        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()

        # Iterate over detected faces and crop them
        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > self.face_confidence_threshold:
                bbox = predictions[0, 0, i, 3:7] * np.array(
                    [self.w, self.h, self.w, self.h]
                )
                (startX, startY, endX, endY) = bbox.astype("int")

                # Calculate the center, width, and height of the bounding box
                centerX = (startX + endX) // 2
                centerY = (startY + endY) // 2
                width = endX - startX
                height = endY - startY

                # Scale dimensions based on scaleFactor
                newWidth = int(width * self.scaleFactor)
                newHeight = int(height * self.scaleFactor)

                # Calculate new coordinates for the scaled bounding box
                newStartX = int(centerX - newWidth // 2)
                newStartY = int(centerY - newHeight // 2)
                newEndX = int(centerX + newWidth // 2)
                newEndY = int(centerY + newHeight // 2)

                # Crop the scaled face region
                cropped_face = self.img[newStartY:newEndY, newStartX:newEndX]
                cropped_faces.append(cropped_face)

        return cropped_faces

    def processVideo_save_faces(
        self, videoName: str, outputFolder: str, nth_frame: int = 1, min_dim: int = 100
    ) -> None:
        """
        Process the given video, extract faces, and save them to the specified folder if they meet the minimum dimension requirement.

        Parameters:
        videoName (str): The path to the video file.
        outputFolder (str): The folder where the cropped faces will be saved.
        nth_frame (int): The interval at which frames will be processed. Default is 1, meaning every frame will be processed.
        min_dim (int): The minimum dimension (both height and width) required for a face to be saved. Default is 50.
        """

        # Open the video file
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print("[ERROR] Opening video stream or file")
            return

        # Create the output folder if it doesn't exist
        os.makedirs(outputFolder, exist_ok=True)

        # Initialize frame count and FPS counter
        frame_count = 0
        failed_frames = 0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = FPS().start()
        framerate = int(cap.get(cv2.CAP_PROP_FPS))

        # Loop through video frames
        with tqdm(total=num_frames, desc="Processing Frames") as pbar:
            while True:
                success, self.img = cap.read()
                frame_count += 1
                pbar.update(1)

                # Handle unsuccessful frame reads
                if not success and frame_count < num_frames:
                    failed_frames += 1
                    continue
                elif not success and frame_count >= num_frames:
                    print(f"[INFO] Finished processing all frames for {videoName}")
                    break

                # Process frame
                self.h, self.w = self.img.shape[:2]
                cropped_faces = self.extractFace()

                # Save the extracted faces from the nth frame
                if frame_count % nth_frame == 0:
                    for idx, face in enumerate(cropped_faces):
                        if face is not None and face.size > 0:
                            face_height, face_width = face.shape[:2]
                            if face_height >= min_dim and face_width >= min_dim:
                                face_filename = os.path.join(
                                    outputFolder,
                                    f"frame_{frame_count:04d}_face_{idx}.jpg",
                                )
                                cv2.imwrite(face_filename, face)

                # Update FPS counter
                fps.update()

        # Release video resources
        fps.stop()
        cap.release()

        # Display processing statistics
        print(f"[INFO] Frames processed per second: {fps.fps():.2f}")
        print(f"[INFO] Processing time: {fps.elapsed() / 60:.2f} minutes")
        print(f"[INFO] Video Length: {num_frames / framerate / 60:.2f} minutes")
        print(f"[INFO] Video framerate: {framerate:.2f} fps")
        print(
            f"[INFO] Failed frames: {failed_frames} Percentage: {failed_frames/num_frames*100:.2f}%"
        )
