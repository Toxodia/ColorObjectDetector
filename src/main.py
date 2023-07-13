# Python code for Orange Color Detection
import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)
trigger = True


def find_minimum_y(array):
    if len(array) == 0:
        return None  # Return None or any other default value when the array is empty

    flattened_array = np.concatenate(array)
    y_values = flattened_array[:, 0, 1]  # Extract y-values from each [[x, y]] pair

    if len(y_values) == 0:
        return None  # Return None or any other default value when no y-values are found

    min_y = np.min(y_values)
    return min_y


while trigger:
    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for orange color and define mask
    orange_lower = np.array([5, 75, 180], np.uint8)
    orange_upper = np.array([36, 163, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Transform image to only detect pixels within mask range
    kernel = np.ones((5, 5), "uint8")
    orange_mask = cv2.dilate(orange_mask, kernel)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame,
                                 mask=orange_mask)

    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(orange_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # print highest pixel with color for later usage
    print(find_minimum_y(contours))

    # Draw Box around orange groups
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 165, 255), 2)

            cv2.putText(imageFrame, "Orange Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 165, 255))

    # Title for window
    cv2.imshow("Orange Color Detection in Real-TIme", imageFrame)

    # Program Termination
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # cap.release()
        cv2.destroyAllWindows()
        trigger = True
        break
