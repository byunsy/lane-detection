"""============================================================================
TITLE      : lane_detection.py
BY         : Sang Yoon Byun
DESCRIPTION: A simple program that can detect lanes in a video using 
             edge-detection methods (Canny and Bilaterial Filter)
============================================================================"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

"""============================================================================
PROCEDURE:
    canny
PARAMETERS:
    image, an image
PURPOSE:
    detects a wide range of edges in images using the Canny edge method
PRODUCES:
    canny, a binarized image with various edges (white) detected
============================================================================"""
def canny(image):

    # Convert image grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image using bilateral filter
    blur = cv2.bilateralFilter(gray, -1, 10, 5)

    # Extract edges using Canny method
    canny = cv2.Canny(blur, 50, 150)

    return canny


"""============================================================================
PROCEDURE:
    region_of_interest
PARAMETERS:
    image_h_w, height and width of the image
PURPOSE:
    create a mask for a specific region of interest
PRODUCES:
    mask, a mask where the region of interest is 255 and the rest are 0
============================================================================"""
def region_of_interest(image_h_w):

    # Create a black image with same dimensions
    mask = np.zeros(image_h_w, np.uint8)

    # Attain height and width information of image
    img_h, img_w = image_h_w

    # Set specific points
    pts = np.array([[int(0.45 * img_w), int(img_h*0.6)],
                    [int(0.55 * img_w), int(img_h*0.6)],
                    [int(0.90 * img_w), img_h],
                    [int(0.10 * img_w), img_h]])

    #    ______________
    #   |     ____     |
    #   |    /    \    |
    #   |   /      \   |
    #   |  /________\  |  current ROI looks like this

    # Show the region of interest
    # cv2.polylines(image, [pts], True, (255, 0, 255), 1)

    # Fill the region with white pixels
    cv2.fillPoly(mask, [pts], 255)

    return mask


"""============================================================================
PROCEDURE:
    display_line
PARAMETERS:
    image_h_w, height and width of the image
    lines, an np-array of all the lines in (x1, y1, x2, y2) coordinate format
      [[[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ... [[x1, y1, x2, y2]]]
PURPOSE:
    draws lines on a blank black image based on the lines parameter given
PRODUCES:
    detected_linse, an image with lines drawn on a black background
============================================================================"""
def display_line(image_h_w, lines):

    # Create a black image with same dimensions
    detected_lines = np.zeros(image_h_w, np.uint8)

    # Convert grayscale to BGR (so that colored lines will show)
    detected_lines = cv2.cvtColor(detected_lines, cv2.COLOR_GRAY2BGR)

    # For every line detected
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            cv2.line(detected_lines, pt1, pt2, (255, 255, 0), 10, cv2.LINE_AA)

    return detected_lines


"""============================================================================
PROCEDURE:
    make_coordinates
PARAMETERS:
    img_height, height of an image
    line_parameters, a list containing slope and y-intercept of a line
      [slope, y_intercept]
PURPOSE:
    computes appropriate coordinates for lines that can be drawn on an image
    using the slope and y-intercept information
PRODUCES:
    coord, an np-array of computed coordinates 
============================================================================"""
def make_coordinates(img_height, line_parameters):

    slope, y_intercept = line_parameters

    y1 = img_height
    y2 = int(y1 * 0.6)

    # y = mx + b --> x = (y-b)/m
    x1 = int((y1 - y_intercept) / slope)
    x2 = int((y2 - y_intercept) / slope)

    return np.array([x1, y1, x2, y2])


"""============================================================================
PROCEDURE:
    avg_slope_intercept
PARAMETERS:
    img_height, height of an image
    lines, an np-array of all the lines in (x1, y1, x2, y2) coordinate format
      [[[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ... [[x1, y1, x2, y2]]]
    left_fit_p, left_fit values from the previous cycle
    right_fit_p, right_fit values from the previous cycle
PURPOSE:
    calculates the average slope and average y-intercept for left and right
PRODUCES:
    lr_lines, an np-array of coordinates for two lines
      [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    left_fit, slopes and y-intercepts calculated from this cycle
    right_fit, slopes and y-intercepts calculated from this cycle
============================================================================"""
def avg_slope_intercept(img_height, lines, left_fit_p, right_fit_p):

    # Two lists to hold information about the lines on the left and right
    # Each will hold slopes and y-intercepts [(slope, y-intercept),...]
    left_fit = []
    right_fit = []

    # For each line
    if lines is not None:
        for i in range(lines.shape[0]):

            # Calculates slope and y-intercept
            parameters = np.polyfit((lines[i][0][0], lines[i][0][2]),
                                    (lines[i][0][1], lines[i][0][3]), 1)
            slope = parameters[0]
            y_intercept = parameters[1]

            if slope > 0:  # lines on the left will have positive slopes
                left_fit.append((slope, y_intercept))
            else:         # lines on the right will have negative slopes
                right_fit.append((slope, y_intercept))

    # If either list is empty, use the previous values (from params given)
    if not left_fit:
        left_fit = left_fit_p    # values from prev cycle
    elif not right_fit:
        right_fit = right_fit_p  # values from prev cycle

    # Calculate mean slope and mean y-intercept for left and right fits
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # Use the means to calculate coordinates for a single line
    left_line = make_coordinates(img_height, left_fit_avg)
    right_line = make_coordinates(img_height, right_fit_avg)

    return np.array([left_line, right_line]), left_fit, right_fit


"""============================================================================
                                     MAIN
============================================================================"""
def main():

    # Create class object of VideoCapture
    cap = cv2.VideoCapture("./test_vid/test_video3.mp4")

    # Check for any errors opening the video
    if not cap.isOpened():
        print("Error: Failed to open video.")
        sys.exit()

    # Initialize left_fit and right_fit (will be used later)
    # - left_fit_1 is the previous fit; left_fit_2 is the current fit
    # - Same for right_fit_1 and right_fit_2
    left_fit_1 = [0, 0]
    right_fit_1 = [0, 0]

    while True:

        # Keep reading the frames
        ret, frame = cap.read()

        # Exit when video finishes
        if not ret:
            print("Video has ended.")
            sys.exit()

        # Make a copy of source image
        cpy = np.copy(frame)

        # Attain height and width information of image
        img_h, img_w = frame.shape[:2]
        img_h_w = frame.shape[:2]

        # Extract edges of image using Canny method
        edge_img = canny(frame)
        cv2.namedWindow("Edge", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Edge', img_w//2, img_h//2)  
        cv2.imshow("Edge", edge_img)                  

        # Create mask for region of interest
        mask = region_of_interest(img_h_w)

        # Isloate the region of interest using mask
        edge_road = cv2.bitwise_and(edge_img, mask)

        # Detect lines using Hough transform technique
        lines = cv2.HoughLinesP(edge_road, 2.0, np.pi/180.0, 100,
                                minLineLength=15, maxLineGap=5)

        # Display all detected lines on black image
        detected_lines = display_line(img_h_w, lines)
        cv2.namedWindow("Lines", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lines', img_w//2, img_h//2)
        cv2.imshow("Lines", detected_lines)

        # Calculate a single averaged line on the left and right
        avg_lines, left_fit_2, right_fit_2 = avg_slope_intercept(img_h,
                                                                 lines,
                                                                 left_fit_1,
                                                                 right_fit_1)

        # Display the two averaged lines on black image
        lanes = display_line(img_h_w, avg_lines)

        # Blend it with the original source image
        lanes_on_img = cv2.addWeighted(cpy, 0.95, lanes, 1, 1)

        # Show
        cv2.namedWindow("Detected Lanes")
        cv2.imshow("Detected Lanes", lanes_on_img)

        # Breaking out of the loop
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

        # Current left_fit and right_fit now becomes the previous ones
        # Goes back to top of the loop to get new current fits
        left_fit_1 = left_fit_2
        right_fit_1 = right_fit_2

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
