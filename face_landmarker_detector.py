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


def overlay_image_rotated(background, overlay, left_eye, right_eye):
    # Compute center, angle, and scale between eyes
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(-eye_dy, eye_dx))
    eye_distance = np.sqrt(eye_dx ** 2 + eye_dy ** 2)

    # Desired overlay width relative to eye distance
    desired_width = int(eye_distance * 1.7)
    scale = desired_width / overlay.shape[1]
    new_size = (desired_width, int(overlay.shape[0] * scale))

    # Resize overlay
    overlay_resized = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

    # Compute rotation matrix
    center = (overlay_resized.shape[1] // 2, overlay_resized.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay_rotated = cv2.warpAffine(overlay_resized, rot_matrix, (overlay_resized.shape[1], overlay_resized.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Compute placement point
    mid_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    top_left = (mid_eye[0] - overlay_rotated.shape[1] // 2, mid_eye[1] - overlay_rotated.shape[0] // 2)

    # Overlay with alpha blending
    x, y = top_left
    h, w = overlay_rotated.shape[:2]

    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background  # Out of bounds

    alpha_overlay = overlay_rotated[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay

    for c in range(3):  # BGR channels
        background[y:y + h, x:x + w, c] = (
                alpha_overlay * overlay_rotated[:, :, c] +
                alpha_bg * background[y:y + h, x:x + w, c]
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

decoration = cv2.imread('glasses2.png', cv2.IMREAD_UNCHANGED)  # Shape: (H, W, 4)


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

    # if detection_result.face_landmarks:
    #     landmarks = detection_result.face_landmarks[0]
    #
    #     # Use landmark 33 and 263 to anchor the decoration (eye corners)
    #     left_eye = landmarks[33]
    #     right_eye = landmarks[263]
    #
    #     x1 = int(left_eye.x * frame.shape[1])
    #     y1 = int(left_eye.y * frame.shape[0])
    #     x2 = int(right_eye.x * frame.shape[1])
    #     y2 = int(right_eye.y * frame.shape[0])
    #
    #     # Determine overlay position and size
    #     center_x = (x1 + x2) // 2
    #     center_y = (y1 + y2) // 2
    #     width = int(1.6 * abs(x2 - x1))
    #     height = int(width * decoration.shape[0] / decoration.shape[1])  # keep aspect ratio
    #
    #     top_left_x = center_x - width // 2
    #     top_left_y = center_y - height // 2
    #
    #     annotated_image = overlay_image(annotated_image, decoration, top_left_x, top_left_y, (width, height))
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        frame_h, frame_w, _ = frame.shape

        left_eye = landmarks[33]
        right_eye = landmarks[263]

        left_eye_coords = (int(left_eye.x * frame_w), int(left_eye.y * frame_h))
        right_eye_coords = (int(right_eye.x * frame_w), int(right_eye.y * frame_h))

        annotated_image = overlay_image_rotated(annotated_image, decoration, left_eye_coords, right_eye_coords)

    cv2.imshow('result', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()