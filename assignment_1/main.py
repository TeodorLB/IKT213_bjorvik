import cv2
import numpy as np
import os

def print_image_information(image):
    image = cv2.imread(image)
    height, width, channels = image.shape
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {image.size}")
    print(f"Data type: {image.dtype}")

def save_camera_info():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Use camera index 1
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read a frame to ensure the camera is initialized
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}, Width: {width}, Height: {height}")  # Debug output
    print("It doesn't seem like the cameras values are being read correctly. Displaying the video feed works, which makes me thing my camera does not support these properties being read. The code should work on other cameras.")

    output_dir = os.path.expanduser("assignment_1/solutions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "camera_outputs.txt")

    with open(output_file, "w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"Camera information saved to {output_file}")

    cap.release()

if __name__ == "__main__":
    print_image_information("assignment_1/lena.png")
    save_camera_info()




