# lane-detection

## Description

Lane-detection is a simple program that can detect and identify lanes within a video in real time. The project contains fundamental computer vision concepts and ideas about lane detection in an autonomous vehicle. More advanced concepts are handled in a separate repository called "enhanced-lane-detection". As the user passes in an input video, the program will automatically identify the current lane in blue.

The objective of lane-detection project was to learn the basics process of perception for self-driving cars. This project involves reading an input video, frame by frame, and applying Canny and Bilateral filters to accurately detect edges within the frame. Users can set their own regions of interest to select the specific areas to focus on. The edge-detection method performs well in clear, clean, and straight roads, but have significant drawbacks in situations where the roads are uneven, lanes are faded, and tracks are covered in shadows.

The lanes are calculated first by accumulating all detected lines of significant length in the region of interest. The lines are then organized into left and right lines based on their slopes (left lines will naturally have positive slopes and right lines will have negative slopes). Afterwards, the program computes the average slope and y-intercept for these left and right lines to generate two final lines for the detected lane.

## Installation

I used the OpenCV package for python (version 4.1.0.25 or above) with Python 3.7.2

```bash
pip install opencv-python==4.1.0.25
```

## Usage

Clone the lane-detection repository in your directory.

```bash
git clone https://github.com/byunsy/lane-detection.git
```

Move to your specific directory and execute the program.

```bash
python lane-detection.py
```

## Demonstrations

Test Case 1.
![](images/lane_detection1.gif)

Test Case 2.
![](images/lane_detection2.gif)
