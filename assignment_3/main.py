import cv2
import numpy as np

def gGblur(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0) # width height sigmaX/Y
    return img_blur

def sobel_edge_detection(image):
    img_blur = gGblur(image)
    sobelxy = cv2.Sobel(img_blur, cv2.CV_64F, 1, 1, ksize=1) # dx dy ksize dont think i need to specify argument name, just if order of arguments is non standard?

    cv2.imwrite("assignment_3/outputs/sobel_edge_detection.png", sobelxy)

def canny_edge_detection(image, threshold1, threshold2):
    img_blur = gGblur(image)
    canny = cv2.Canny(img_blur, threshold1, threshold2) #threshold1 threshold2

    cv2.imwrite("assignment_3/outputs/canny_edge_detection.png", canny)

def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch collumns and rows
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    
    cv2.imwrite("assignment_3/outputs/res.png", image)

def resize(image, scale_factor, up_or_down):
    # Round down to nearest power of 2 (1, 2, 4, 8, ...) (since pyramid scaling isnt for specific factors like 1.5 or 3 etc)
    if scale_factor < 1:
        scale_factor = 1
    else:
        scale_factor = 2 ** int(np.floor(np.log2(scale_factor)))
    num_steps = int(np.log2(scale_factor))
    for _ in range(num_steps):
        if up_or_down == "up":
            image = cv2.pyrUp(image)
        elif up_or_down == "down":
            image = cv2.pyrDown(image)
    cv2.imwrite("assignment_3/outputs/resized.png", image)


if __name__ == "__main__":
    lambo = cv2.imread("assignment_3/inputs/lambo.png")
    if lambo is None:
        print("Image not found.")
    else:
        #1
        sobel_edge_detection(lambo)

        #2
        canny_edge_detection(lambo,50,50) #threshold1, threshold2

        #4
        resize(lambo,2,"up") #Function arguments: image, scale_factor: int (should be power of 2, but is rounded down to closest power of 2 to ensure execution), up_or_down: str ("up" or "down")
    
    shapes = cv2.imread("assignment_3/inputs/shapes-1.png")
    shapes_template = cv2.imread("assignment_3/inputs/shapes_template.jpg")
    if shapes is None or shapes_template is None:
        print("Image not found.")
    else:
        #3
        template_match(shapes, shapes_template)

    
    