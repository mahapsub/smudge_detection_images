import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import math

def main():
    image_lst = parse_data()
    # limg = image_lst[0]
    # rimg = image_lst[1]
    # # display_image([limg, rimg])
    # print(limg.shape)
    # print(rimg.shape)
    # img = rgb2gray(limg)
    # flat_img = img.flatten()
    # print(len(flat_img))
    # dim = int(math.sqrt(len(flat_img)))


    # plt.imshow(flat_img.reshape((dim,-1)), cmap="gray")
    # plt.show()
    threshold_img = np.zeros((2032,2032))
    print(threshold_img.shape)
    print(type(threshold_img))
    for i in range(len(image_lst)):
        img = rgb2gray(image_lst[i])
        # print(type(img))
        # print(img.shape)
        threshold_img +=img
    # threshold_img = int(threshold_img/len(image_lst))
    threshold_img = cv2.equalizeHist(threshold_img)
    kernel = np.ones((50,50),np.float32)/(50*50)
    thresh = cv2.filter2D(threshold_img,-1,kernel)
    thresh = cv2.medianBlur(np.float32(thresh),5)
    # thresh = cv2.threshold(threshold_img, 110, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(threshold_img,100,cv2.ADAPTIVE_THRESH_MEAN_C,\
            # cv2.THRESH_BINARY,11,2)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)


    plt.imshow(thresh, cmap="gray")
    plt.show()
    
        
    

def rgb2gray(rimg):
    return cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
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

    dir_path = '/Users/mahapsub/workspace/msai/geospatial/data/sample_drive/cam_3/'
    print('reading images')
    image_names = [file for file in glob.glob(dir_path+"*.jpg")]
    sample_size = 0.05
    sz = int(len(image_names)*sample_size)
    print('size of images = ',sz)
    sample_images = image_names[0:sz]
    # print('image names:{}'.format([name for name in sample_images]))

    sample_lst = [cv2.imread(file) for file in sample_images]
    print('done reading')

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
    