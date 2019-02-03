# Pop_Eye (Kriti 2018)
The computer vision subsystem for an autonomous robotic arm undertaken in Kriti 2018.

## Libraries and Dependencies-
* OpenCV3
* Numpy
* SciPy
* PySerial

## Working
* ### calib(img, flag)
    * Perform calibration for a single frame
    * Find image centre and map pixels to centimeters
* ### round1(t)
    * Set thresholds to detect red color
    * Find contours in the mask and find the largest one (red square)
    * Find arm head current location
    * Find the r and theta for the red square
    * Move arm to the location
* ### detect(color, frame)
    * Set thresholds according to color
    * Convert to HSV
    * Form threshold mask and apply blurring to minimize noise
    * Apply Hough circle transform to detect circles
    * Run for 50 frames
    * Return the location of balloons
* ### dist(x1, y1, x2, y2)
    * Return distance between (x1,y1) and (x2,y2)
* ### angle(ox, oy, x1, y1, x2, y2)
    * Return angle between (x1, y1), (ox,oy) and (x2, y2)
