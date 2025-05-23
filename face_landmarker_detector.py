'''
Download the face_landmarker_v2_with_blendshapes task file from the MediaPipe model zoo.
!wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
'''
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def overlay_image(background, overlay, x, y, overlay_size):
    overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background  # Don't draw outside bounds

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):  # For BGR channels
        background[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y+h, x:x+w, c]
        )

    return background


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

decoration = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)  # Shape: (H, W, 4)


while True:
    # STEP 3: Load the input image.
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    annotated_image = np.copy(image.numpy_view()) # draw_landmarks_on_image(image.numpy_view(), detection_result)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]

        # Use landmark 33 and 263 to anchor the decoration (eye corners)
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        x1 = int(left_eye.x * frame.shape[1])
        y1 = int(left_eye.y * frame.shape[0])
        x2 = int(right_eye.x * frame.shape[1])
        y2 = int(right_eye.y * frame.shape[0])

        # Determine overlay position and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = int(1.6 * abs(x2 - x1))
        height = int(width * decoration.shape[0] / decoration.shape[1])  # keep aspect ratio

        top_left_x = center_x - width // 2
        top_left_y = center_y - height // 2

        annotated_image = overlay_image(annotated_image, decoration, top_left_x, top_left_y, (width, height))

    cv2.imshow('result', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()