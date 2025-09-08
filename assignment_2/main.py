import cv2
import numpy as np

def padding(image, border_width):
    padded_img = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    return padded_img

def crop(image, x_0, x_1, y_0, y_1):
    cropped = img[x_0:x_1, y_0:y_1]
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def copy(image, emptyPictureArray):
            height, width, channels = image.shape
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        emptyPictureArray[y, x, c] = image[y, x, c]
            return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = (int(image[y, x, c]) + int(hue)) % 256
    shifted_image = emptyPictureArray
    return shifted_image

def smoothing(image):
    smoothed = cv2.GaussianBlur(image,(15, 15), 0,0,borderType=cv2.BORDER_DEFAULT)
    return smoothed

def rotation(image, angle):
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    return rotated

if __name__ == "__main__":
    img = cv2.imread("assignment_2/lena-2.png")
    if img is None:
        print("Image not found.")
    else:
        print(img.shape)
        padded = padding(img, 100)
        cv2.imwrite("assignment_2/lena_padded.png", padded)

        cropped = crop(img, 80, 512-80, 130, 512-130)
        cv2.imwrite("assignment_2/lena_cropped.png", cropped)

        rezised = resize(img,200,200)
        cv2.imwrite("assignment_2/lena_resized.png", rezised)

        height, width, channels = img.shape
        emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
        copied = copy(img, emptyPictureArray)
        cv2.imwrite("assignment_2/lena_copied.png", copied)

        grayscaled = grayscale(img)
        cv2.imwrite("assignment_2/lena_grayscaled.png", grayscaled)

        hsved = hsv(img)
        cv2.imwrite("assignment_2/lena_hsv.png", hsved)

        emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
        colorshifted = hue_shifted(img, emptyPictureArray, 50)
        cv2.imwrite("assignment_2/lena_hue_shifted.png", colorshifted)

        smoothed = smoothing(img)
        cv2.imwrite("assignment_2/lena_smoothed.png", smoothed)

        rotated = rotation(img, 180)
        cv2.imwrite("assignment_2/lena_rotated.png", rotated)