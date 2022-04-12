import os
import cv2
import numpy as np
from math import cos, sin


def remove_glare(image):
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(H[150:])
    # plt.show()
    idx = np.argmax(H[150:]) + 151
    binary = cv2.threshold(image, idx, 255, cv2.THRESH_BINARY)[1]

    st3 = np.ones((3, 3), dtype="uint8")
    st7 = np.ones((7, 7), dtype="uint8")

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, st3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, st3, iterations=2)

    im_floodfill = binary.copy()

    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, st7, iterations=1)
    _, _, stats, cents = cv2.connectedComponentsWithStats(im_out)
    cx, cy = 0, 0
    for st, cent in zip(stats, cents):
        if 1500 < st[4] < 3000:
            if 0.9 < st[2] / st[3] < 1.1:
                cx, cy = cent.astype(int)
                r = st[2] // 2
                cv2.circle(image, (cx, cy), r, (125, 125, 125), thickness=2)

    image = np.where(im_out, 80, image)
    image = cv2.medianBlur(image, 5)

    return image, cx, cy

def exploding_circle(image, cx, cy, step_seed_point = 15, step_radius = 5, min_radius = 20, max_radius = 100, step_angle = 10):

    max_diff = 0
    best_x = 0
    best_y = 0
    best_radius = 0

    for dx in range(-1,2):
        for dy in range(-1,2):
            x1 = cx + step_seed_point * dx
            y1 = cy + step_seed_point * dy

            previous_brighness = None

            for radius in range(min_radius,max_radius,step_radius):

                brightness_sum = 0


                for angle in range(0,360,step_angle):
                    
                    x2 = int(radius * cos(angle) + x1)
                    y2 = int(radius * sin(angle) + y1)
                    
                    #if x2>
                    brightness_sum += image[x2, y2]

                
                image_circle = np.copy(image)
                cv2.circle(image_circle,(x1,y1),radius,(255, 0, 0))
                cv2.imshow("circle", image_circle)
                key = cv2.waitKey()          

                
                average_brightness = brightness_sum / ((360-0)/ step_angle)

                if previous_brighness != None:
                    if max_diff<abs(previous_brighness-average_brightness):
                        max_diff = abs(previous_brighness-average_brightness)
                        best_x = x1
                        best_y = y1
                        best_radius = radius
                
                previous_brighness = average_brightness

    image_circle = np.copy(image)
    cv2.circle(image_circle,(best_x,best_y),best_radius,(255, 0, 0))
    cv2.imshow("best circle", image_circle)
    key = cv2.waitKey()   

    return 0




def main(data_path):
    # Get files from data path
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]

    for filename in filename_list:
        # Read image
        img = cv2.imread(os.path.join(data_path, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        exploding_circle(img_no_glare,x,y)

        # Gabor filters
        # TODO

        cv2.imshow("Original image", img)
        cv2.imshow("Gray", gray)
        cv2.imshow("No glare", img_no_glare)
        key = cv2.waitKey()
        if key == ord("x"):
            break


if __name__ == "__main__":
    data_path = "./iris_database_train"
    main(data_path)

