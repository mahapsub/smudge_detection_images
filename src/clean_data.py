import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import math
import copy

def main():
    images = parse_data()
    if len(images)==0:
        print('No images found')
        return False
    thresh = np.zeros((2032,2032,3))

    if len(images)> 80:
        image_lst = images[:80]
    else:
        image_lst = images

    #Iterate through sample images and generate a mean image
    for i in range(len(image_lst)):
        img = image_lst[i]
        thresh += img
    thresh = thresh/len(image_lst)
    
    

    thresh = np.array(np.round(thresh),dtype=np.uint8)
    #plot1
    mean = thresh

    median_blur_size = 3
    #Blur added here
    thresh = cv2.medianBlur(thresh,median_blur_size)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    #plot2
    bw_medianblur = thresh

    #Thresholding techniques
    _,thresh1 = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,107,13)

    # blur = cv2.GaussianBlur(thresh,(5,5),0)
    # ret3,thresh4 = cv2.threshold(blur,105,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #Select everything that is not thresholded (i.e find the mask)
    mask = cv2.bitwise_not(thresh2)

    #plot3


    mask_found = mask
    # contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]

    #Determine the size and location of the mask
    M = cv2.moments(mask)

    #Uses the entire folder to find a random image
    trial_img = images[np.random.randint(0,len(images))]
    #plot4 
    random_orig = copy.deepcopy(trial_img)

    #moo checks the size of the blur.
    if check_size(M):
    # if 'm00' in M:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # put text and highlight the center


        cv2.circle(mask, (cX, cY), 75, (100, 100, 100), 5)
        cv2.putText(mask, "Smudge", (cX - 120, cY ),cv2.FONT_HERSHEY_SIMPLEX, 2, (145, 76, 0), 4)
        #Applying mask
        cv2.circle(trial_img, (cX, cY), 75, (100, 100, 100), 5)
        #Applying Text
        cv2.putText(trial_img, "Smudge", (cX - 120, cY ),cv2.FONT_HERSHEY_SIMPLEX, 2, (145, 76, 0), 4)


    #Generate plots. 
    fig, axs = plt.subplots(2, 3, figsize=(50, 30), sharey=True)
    axs[0][0].imshow(mean)
    axs[0][0].set_title('The average of {} imgs'.format(len(image_lst)))
    axs[0][1].imshow(bw_medianblur)
    axs[0][1].set_title('post analysis at {}'.format(median_blur_size))
    axs[0][2].imshow(thresh1)
    axs[0][2].set_title('Simple binary threshold')
    axs[1][0].imshow(mask_found)
    axs[1][0].set_title('Checking to see if mask is found')
    axs[1][1].imshow(random_orig)
    axs[1][1].set_title('Selecting random image from corpora')
    axs[1][2].imshow(trial_img)
    axs[1][2].set_title('Application of mask to random image')
    plt.show()

def check_size(M):
    #Makes sure that the smudge is an appropriate size.
    size = M['m00']
    # print('SIZE', size)
    if size > 50000 and  size < 250000:
        return True
    else:
        return False


def rgb2gray(rimg):
    return rimg
    # return cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    # r = rimg[:,:,0]
    # g = rimg[:,:,0]
    # b = rimg[:,:,0]
    # gray = r*.299 + g*.5870 + b*.1140
    # return gray
    # return np.dot(rimg[...,:3], [0.2989, 0.5870, 0.1140])


def parse_data():
    '''
        Reads in data and parses dataset
    '''
    dir_path = '/Users/mahapsub/workspace/msai/geospatial/smudge_detection_images/data/sample_drive/cam_3/'

    print('reading images from \'{}\''.format(dir_path))
    image_names = [file for file in glob.glob(dir_path+"*.jpg")]
    sample_size = 0.020
    sz = int(len(image_names)*sample_size)
    
    sample_images = image_names[0:sz]
    # print('image names:{}'.format([name for name in sample_images]))

    sample_lst = []
    for i in range(len(sample_images)):
        sample_lst.append(cv2.imread(sample_images[i]))
        if i % (int(len(sample_images)*.10)) == 0:
            print('loaded {0} images out of {1}'.format(i, len(sample_images)))

    # sample_lst = [cv2.imread(file) for file in sample_images]

    return sample_lst

def display_image(lst):
    '''
        Helper method to display data within window
    '''
    for img in lst:
        img = cv2.resize(img, (1280, 700))
        cv2.imshow("image", img)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    