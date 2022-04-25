import os
import cv2
import numpy as np
from math import cos, sin, pi
from skimage.filters import gabor_kernel, gabor
from scipy import ndimage as ndi


class circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
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

def exploding_circle_iris(image, cx, cy, step_seed_point = 3, step_radius = 1, min_radius = 35, max_radius = 150, step_angle = 5):

    max_diff = 0
    best_circle=circle(0,0,0)

    for dx in range(-5,6):
        for dy in range(-5,6):
            x1 = cx + step_seed_point * dx
            y1 = cy + step_seed_point * dy

            previous_brighness = None

            for radius in range(min_radius,max_radius,step_radius):

                brightness_sum = 0

                for angle in range(0,360,step_angle):
                    
                    x2 = int(radius * cos((angle*pi)/180) + x1)
                    y2 = int(radius * sin((angle)*pi/180) + y1)

                    brightness_sum += image[y2,x2]
                
                average_brightness = float(brightness_sum) / ((360-0)/ float(step_angle))

                if previous_brighness != None:
                    if max_diff<(average_brightness-previous_brighness):
                        max_diff =(average_brightness-previous_brighness)
                        best_circle = circle(x1,y1,radius)
                
                previous_brighness = average_brightness


    return best_circle

def exploding_circle_pupil(image, cx, cy, step_seed_point = 3, step_radius = 10, min_radius = 200, max_radius = 400, step_angle = 2, left = True):

    max_diff = 0
    best_circle=circle(0,0,0)

    previous_brighness = None

    if left == True:
        i=1
    else :
        i=0

    for radius in range(min_radius,max_radius,step_radius):
        brightness_sum = 0

        image_copie=image.copy()

        for angle in range(-45+180*i,45+180*i,step_angle):
            
            x2 = int(radius * cos((angle*pi)/180) + cx)
            y2 = int(radius * sin((angle)*pi/180) + cy)

            #check that the coordinates are inside the image
            if x2<0 or x2>=image.shape[1] or y2<0 or y2>=image.shape[0]:
                break

            brightness_sum += image[y2,x2]
        
        average_brightness = float(brightness_sum) / ((120)/ float(step_angle))

        if previous_brighness != None:
            if max_diff<(average_brightness-previous_brighness):
                max_diff =(average_brightness-previous_brighness)
                best_circle = circle(cx,cy,radius-step_radius)
        
        previous_brighness = average_brightness


    return best_circle

#Gabor filter extration from the image fonction
def gabor_filter(image, circle_iris, left_iris_circle, right_iris_circle, step_radius=0.075):

    kernels = []
    sigma = 2
    for theta in range(8):
        t = theta / 8. * np.pi
        kernel = (gabor_kernel(0.15, theta=t, sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)

    code=[]
    j=0
    k=0
    #Loop for each radius
    image_copy=image.copy()
    for i in range(8):

        radius_left = circle_iris.r + step_radius * (i+1) * (left_iris_circle.r - circle_iris.r)
        #Loop for each angle left

        for angle_i in range(8):
            angle=(angle_i/8)*(pi/2)+(3*pi)/4
            j=j+1
            x2 = int(radius_left * cos(angle) + circle_iris.x)
            y2 = int(radius_left * sin(angle) + circle_iris.y)
            image_copy[y2,x2]=255
            cv2.imshow("image_copy",image_copy)
            cv2.imshow("analyse-place",image[y2-10:y2+10,x2-10:x2+10])
            cv2.waitKey()
            for gabor_filter in kernels:
                data=(ndi.convolve(image[y2-10:y2+10,x2-10:x2+10], gabor_filter,mode='wrap'))
                
                #Sum of the gabor filter
                sum=np.sum(data)
                k=k+2
                if sum.real>0:
                    code.append(1)
                else:
                    code.append(0)
                if sum.imag>0:
                    code.append(1)
                else:
                    code.append(0) 
        
        radius_right=circle_iris.r + step_radius * (i+1) * (right_iris_circle.r - circle_iris.r)
        #Loop for right angle
        for angle_i in range(8):
            angle=(angle_i/8)*(pi/2)-pi/4
            j=j+1
            x2 = int(radius_right * cos(angle) + circle_iris.x)
            y2 = int(radius_right * sin(angle) + circle_iris.y)
            image_copy[y2,x2]=255
            cv2.imshow("image_copy",image_copy)
            cv2.imshow("analyse-place",image[y2-10:y2+10,x2-10:x2+10])
            cv2.waitKey()

            for gabor_filter in kernels:
                data=(ndi.convolve(image[y2-10:y2+10,x2-10:x2+10], gabor_filter,mode='wrap'))
                
                #Sum of the gabor filter
                sum=np.sum(data)
                k=k+2
                if sum.real>0:
                    code.append(1)
                else:
                    code.append(0)
                if sum.imag>0:
                    code.append(1)
                else:
                    code.append(0)                    

    return code


#Function to compare two codes
def compare(code1,code2):

    same=0
    for i in range(len(code1)):
        if code1[i]==code2[i]:
            same+=1
    
    return same/len(code1)


def main(data_path):
    # Get files from data path
    code_dict={}
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]

    for filename in filename_list:
        print("Analysing image: " + filename)
        # Read image
        img = cv2.imread(os.path.join(data_path, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        pupil_circle = exploding_circle_iris(img_no_glare,x,y)
        left_iris_circle = exploding_circle_pupil(gray,x,y,min_radius=pupil_circle.r+50)
        right_iris_circle = exploding_circle_pupil(gray,x,y,min_radius=pupil_circle.r+50,left=False)

        # Draw circles
        cv2.circle(img, (pupil_circle.x, pupil_circle.y), pupil_circle.r, (0, 255, 0), thickness=2)
        img_left=img.copy()
        cv2.circle(img_left, (left_iris_circle.x, left_iris_circle.y), left_iris_circle.r, (0, 255, 0), thickness=2)
        img_right=img.copy()
        cv2.circle(img_right, (right_iris_circle.x, right_iris_circle.y), right_iris_circle.r, (0, 255, 0), thickness=2)

        # Gabor filters
        code_dict[filename]= gabor_filter(gray, pupil_circle, left_iris_circle, right_iris_circle)

    #Compare two image with the same iris
    for i in range(len(code_dict)):
        for j in range(i+1,len(code_dict)):
            print(str(compare(code_dict[filename_list[i]],code_dict[filename_list[j]]))+" "+filename_list[i]+" "+filename_list[j])

if __name__ == "__main__":
    data_path = "./iris_database_train"
    main(data_path)

