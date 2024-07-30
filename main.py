import numpy as np
import json
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from database import save_to_database
import os


def get_weather_weatherapi(city_name, api_key):
    base_url = "http://api.weatherapi.com/v1/current.json?"
    complete_url = base_url + "key=" + api_key + "&q=" + city_name

    response = requests.get(complete_url)
    data = response.json()

    if response.status_code != 200:
        print(f"Error fetching weather data: {data.get('error', {}).get('message', 'Unknown error')}")
        return None

    if "current" in data:
        current = data["current"]
        weather_info = {
            "city": city_name,
            "temperature": current["temp_c"],
            "pressure": current["pressure_mb"],
            "humidity": current["humidity"],
            "description": current["condition"]["text"]
        }
        return weather_info
    else:
        print("Weather data not found in the response.")
        return None


# def randomize_function(road, roadHeat, inputPattern):
#     vehicle_positions = []  # Initialize an empty list for storing positions
#
#     # Randomly choose the pattern type for all lanes
#     pattern_type = inputPattern
#     # pattern_type = 3
#     min_threshold = 120
#     max_threshold = 160
#     for lane in road:
#
#         if pattern_type == 1:
#             # Density gradually decreases in all lanes (most crowded at the start)
#             min_threshold = int(min_threshold / 1.1)  # Ensure integer division
#             max_threshold = int(max_threshold / 1.25)  # Ensure integer division
#
#         elif pattern_type == 2:
#             # All lanes are fully empty, add random noise similar to roadHeat
#             for section in lane:
#                 (x_start, y_start), (x_end, y_end) = section
#                 for i in range(y_start, y_end):
#                     for j in range(x_start, x_end):
#                         if np.random.rand() < roadHeat:
#                             vehicle_positions.append((i, j))
#
#         elif pattern_type == 3:
#             # Maximum crowded in all lanes from start to end
#             min_threshold = 150
#             max_threshold = 160
#
#         for section in lane:
#             num_vehicles_per_section = random.randint(min_threshold, max_threshold)
#
#             (x_start, y_start), (x_end, y_end) = section
#             x_positions = np.random.randint(x_start, x_end, num_vehicles_per_section)
#             y_positions = np.random.randint(y_start, y_end, num_vehicles_per_section)
#             vehicle_positions.extend(list(zip(y_positions, x_positions)))  # Extend with list of tuples
#
#     # print(pattern_type)
#     # print(np.array(vehicle_positions))
#     return vehicle_positions  # Convert to NumPy array and return
def randomize_function():
    global road
    vehicle_positions = []  # Initialize an empty list for storing positions

    # Randomly choose the pattern type for all lanes
    pattern_type = random.choice([1, 2, 3])
    # pattern_type = 2
    min_threshold = 80
    max_threshold = 150

    for lane in road:
        flag = False
        if pattern_type == 1:
            # Density gradually decreases in all lanes (most crowded at the start)
            min_threshold = int(min_threshold / 1.1)  # Ensure integer division
            max_threshold = int(max_threshold / 1.25)  # Ensure integer division

        elif pattern_type == 2:
            # All lanes are fully empty
            for i in range(50):
                vehicle_positions.append((0, 0))
            flag = True

        elif pattern_type == 3:
            # Maximum crowded in all lanes from start to end
            min_threshold = 100
            max_threshold = 250
        if not flag:
            for section in lane:
                num_vehicles_per_section = random.randint(min_threshold, max_threshold)

                (x_start, y_start), (x_end, y_end) = section
                x_positions = np.random.randint(x_start, x_end, num_vehicles_per_section)
                y_positions = np.random.randint(y_start, y_end, num_vehicles_per_section)
                vehicle_positions.extend(list(zip(y_positions, x_positions)))  # Extend with list of tuples

    return vehicle_positions  # Convert to NumPy array and return


def get_time_index():
    '''Get current local time and determine corresponding interval'''
    current_time = datetime.datetime.now()
    hour = current_time.hour + 3 #GMT + 3

    if 4 <= hour < 7:
        return 0  # Early Morning (4:00 AM to 7:00 AM)
    elif 7 <= hour < 10:
        return 1  # Late Morning (7:00 AM to 10:00 AM)
    elif 10 <= hour < 16:
        return 2  # Midday to Early Afternoon (10:00 AM to 4:00 PM)
    elif 16 <= hour < 19:
        return 3  # Late Afternoon (4:00 PM to 7:00 PM)
    elif 19 <= hour < 22:
        return 4  # Evening (7:00 PM to 10:00 PM)
    else:
        return 5  # Night (10:00 PM to 4:00 AM)


def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    return blurred, edges


def create_mask_left(image_shape, line_start, line_end):
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255

    x1, y1 = line_start
    x2, y2 = line_end

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            # Calculate the position relative to the line
            position = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if position > 0:  # Change this condition to switch sides
                mask[y, x] = 0

    return mask


def create_mask_right(image_shape, line_start, line_end):
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255

    x1, y1 = line_start
    x2, y2 = line_end

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            # Calculate the position relative to the line
            position = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if position < 0:  # Change this condition to switch sides
                mask[y, x] = 0

    return mask


def detect_long_lines(edges, min_length, max_gap=10, threshold=50):
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_length, maxLineGap=max_gap)

    long_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > min_length:
                long_lines.append(((x1, y1), (x2, y2)))

    return long_lines


def adjust_clusters_based_on_temperature(current_temp, base_clusters=4):
    # Base number of clusters
    clusters = base_clusters

    # Increase clusters if temperature is high to better differentiate similar heat sources
    if current_temp > 30:
        clusters += 2
    elif current_temp > 40:
        clusters += 3
        takeYellow = False

    return clusters


def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    return blurred, edges


def estimate_cars(area):
    # Define the average area occupied by a car
    average_car_area = 8.1  # Assume 5 square meters

    # Calculate the number of cars
    estimated_cars = area / average_car_area
    return estimated_cars


def main(i, inputPattern):
    global road
    # Step 1: Define Parameters
    image_size = (180, 320)  # Height, Width (wider dimensions)
    grid_size = (18//2, 32//2)  # Grid size for the heatmap

    sidewalk_thickness = 80 # thickness of sidewalk
    road_width = image_size[1] - (2 * sidewalk_thickness) - 20  # in pixels
    beginning_of_road = sidewalk_thickness + 10  # the road begins after sidewalk
    increments = 25
    # num_of_sections = 8 # number of sections per lane
    # section_size = np.linspace(0, image_size[0] - 10, num_of_sections)
    # section_size = int(section_size[1]-section_size[0]) # size of each vertical section in lane

    road = [[((beginning_of_road, 20),      (beginning_of_road+road_width, 60)),
             ((beginning_of_road, 60),      (beginning_of_road+road_width, 100)),
             ((beginning_of_road, 100),      (beginning_of_road+road_width, 140)),
             ((beginning_of_road, 140),      (beginning_of_road+road_width, 180)),
             ]]

    # Step 3: Generate Synthetic Vehicle Positions within Lanes
    vehicle_positions = []

    """**FETCHING REAL-TIME WEATHER CONDITIONS**"""

    # Fetching current weather conditions
    cityName = "jeddah"
    API_KEY = os.getenv("WEATHER_API")
    # currentWeather = get_weather_weatherapi(cityName, API_KEY)
    # currentWeather = currentWeather['temperature']
    currentWeather = 30
    # print(currentWeather)

    """**USING TIME INDEXES**"""

    heat_values = [
        0.02,  # Early Morning (4:00 AM to 7:00 AM)
        0.06,  # Late Morning (7:00 AM to 10:00 AM)
        0.1,   # Midday to Early Afternoon (10:00 AM to 4:00 PM)
        0.07,  # Late Afternoon (4:00 PM to 7:00 PM)
        0.03,  # Evening (7:00 PM to 10:00 PM)
        0.01   # Night (10:00 PM to 4:00 AM)
    ]

    current_time_index = get_time_index()
    # print(f"Current time index is: {current_time_index}")
    current_heat = heat_values[current_time_index]
    # print(heat_values[current_time_index])
    # current_heat = random.choice(heat_values)
    roadHeat = current_heat
    vehicle_positions = randomize_function()

    # Add a car in each possible pixel by a probabilty of .2 to synthesise heat from background
    # for i in range(50):
    #     vehicle_positions.append((0,0))
    for i in range(0, image_size[0]):
        for j in range(0, image_size[1]):
            if np.random.rand() < roadHeat:
            # if j % 20 == 0:
                vehicle_positions.append((i, j))

    vehicle_positions = np.array(vehicle_positions)

    # Display the vehicle positions (for verification)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(vehicle_positions[:, 1], vehicle_positions[:, 0], c='red')
    # plt.xlim(0, image_size[1])
    # plt.ylim(image_size[0], 0)
    # plt.title('Synthetic Vehicle Positions in Lanes')
    # plt.gca().invert_yaxis()
    # # plt.show()

    # Step 4: Calculate the Density Map
    # Initialize a density map
    density_map = np.zeros(grid_size, dtype=np.int32)

    # Calculate cell size
    cell_height = image_size[0] // grid_size[0]
    cell_width = image_size[1] // grid_size[1]

    # Populate density map
    for (y, x) in vehicle_positions:
        grid_x = x // cell_width
        grid_y = y // cell_height
        density_map[grid_y, grid_x] += 1
    # get the average of the density map
    average = np.average(density_map)
    # density_map[0,0] = average
    # Display the density map (for verification)
    # print("Vehicle Density Map:")
    # print(density_map)

    # Step 5: Generate the Heatmap
    # Create a heatmap using seaborn
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(density_map, cmap='hot', linewidths=0.5, annot=False, cbar=True)

    # plt.title('Synthetic Vehicle Density Heatmap')
    # plt.show()

    # Step 6: Overlay Heatmap on a Synthetic Background
    # Create a synthetic background (e.g., a plain gray image)
    background = np.full((image_size[0], image_size[1], 3), 200, dtype=np.uint8)

    # Adding Sidewalks
    sidewalk_color = (128, 128, 128)  # color for sidewalks
    cv2.rectangle(background, (0, 0), (sidewalk_thickness, image_size[0]), sidewalk_color, -1)  # Left sidewalk
    cv2.rectangle(background, (image_size[1] - sidewalk_thickness, 0), (image_size[1], image_size[0]), sidewalk_color, -1)  # Right sidewalk

    # Resize density map to match original image size
    density_map_resized = cv2.resize(density_map.astype('float32'), (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize the density map for visualization
    density_map_resized = (density_map_resized / density_map_resized.max() * 255).astype('uint8')

    # Apply a colormap to the density map
    heatmap = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)

    # Blend the heatmap with the synthetic background
    blended_image = cv2.addWeighted(background, 0.6, heatmap, 0.4, 0)

    # Define points for perspective transformation]
    angle = -0.5
    src_points = np.float32([[0, image_size[0]], [image_size[1], image_size[0]], [0, 0], [image_size[1], 0]])
    dst_points = np.float32([[0, image_size[0]], [image_size[1], image_size[0]], [image_size[1]*0.1, 0], [image_size[1]*(-1*angle), 0]])
    # dst_points = np.float32([[0, image_size[0]],
    #                             [image_size[1], image_size[0]],
    #                             [image_size[1] * (0.3 + angle), image_size[0] * 0.4],
    #                             [image_size[1] * (0.7 + angle), image_size[0] * 0.4]])

    # Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(blended_image, matrix, (image_size[1], image_size[0]))

    # plt.figure(figsize=(15, 10))
    # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    # plt.title('Heatmap Overlay with Angled Perspective')
    # plt.axis('off')
    # plt.show()

    blurred, edges = canny_edge_detection(warped_image[:170, :])
    # Display the original image and the detected edges
    # plt.figure(figsize=(15, 10))
    # plt.subplot(121)
    # plt.imshow(blurred, cmap='gray')
    # plt.title('Blurred Image')
    # plt.axis('off')
    #
    # plt.subplot(122)
    # plt.imshow(edges, cmap='gray')
    # plt.title('Canny Edge Detection')
    # plt.axis('off')
    # plt.show()

    # Function to detect lines and filter based on length

    # Set the minimum length for the lines

    min_length = int(edges.shape[0] * 0.8)  # Adjust this value based on your requirement

    # Detect long lines
    long_lines_rotated = detect_long_lines(edges, min_length, max_gap=20, threshold=50)

    # Print the detected lines
    # for line in long_lines_rotated:
    #     print(f"Start: {line[0]}, End: {line[1]}")
    # Optionally, visualize the result
    output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # for line in long_lines_rotated:
    #     cv2.line(output_img, line[0], line[1], (0, 255, 0), 2)
    # plt.imshow(output_img)
    # plt.title('Detected Long Lines')
    # plt.show()

    # Assume long_lines_rotated is the list of detected lines

    swapped_lines = []

    for line in long_lines_rotated:
        (x1, y1), (x2, y2) = line
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        swapped_lines.append(((x1, y1), (x2, y2)))

    # Print the swapped lines
    # for line in swapped_lines:
    #     print(f"Start: {line[0]}, End: {line[1]}")

    swapped_lines = sorted(swapped_lines, key=lambda x: x[0][0], reverse=False)
    # Display the sorted lines
    # for line in swapped_lines:
    #     print(f"Start: {line[0]}, End: {line[1]}")

    # Assume swapped_lines is the list of detected lines after swapping
    threshold_distance = 70  # Adjust this threshold based on your requirement

    filtered_lines = []
    for i in range(len(swapped_lines)):
        if i == 0:
            filtered_lines.append(swapped_lines[i])
        else:
            (x1_prev, y1_prev), _ = swapped_lines[i - 1]
            (x1_curr, y1_curr), _ = swapped_lines[i]
            distance = np.sqrt((x1_curr - x1_prev)**2 + (y1_curr - y1_prev)**2)
            if distance > threshold_distance:
                filtered_lines.append(swapped_lines[i])

    # # Print the filtered lines
    # for line in filtered_lines:
    #     print(f"Start: {line[0]}, End: {line[1]}")

    # Assume filtered_lines is the list of lines after all previous processing
    # Get the middle x-coordinate of the image
    image_middle_x = edges.shape[1] // 2

    # Initialize variables to store the closest lines
    closest_left_line = None
    closest_right_line = None

    # Initialize minimum distances
    min_left_distance = float('inf')
    min_right_distance = float('inf')

    # Iterate through the filtered lines
    for line in filtered_lines:
        (x1, y1), (x2, y2) = line
        # Calculate the distance from the middle
        if x1 < image_middle_x:
            distance = abs(image_middle_x - x1)
            if distance < min_left_distance:
                min_left_distance = distance
                closest_left_line = line
        elif x1 > image_middle_x:
            distance = abs(image_middle_x - x1)
            if distance < min_right_distance:
                min_right_distance = distance
                closest_right_line = line

    # Return the closest lines
    # Load the image
    image = warped_image.copy()

    # Define the line start and end points
    # line_start = lines[3:, :2].reshape(-1,1)
    # line_end = lines[3:, 2:].reshape(-1,1)  # These points can be adjusted

    # Create a mask to remove the left side
    if closest_left_line is not None:
        mask_left = create_mask_left(image.shape, closest_right_line[0], closest_right_line[1])

    # Create a mask to remove the right side
    if closest_right_line is not None:
        mask_right = create_mask_right(image.shape, closest_left_line[0], closest_left_line[1])

    # Apply the mask to the original image to remove the left side of the line
    cropped_image_left = cv2.bitwise_and(image, image, mask=mask_left)

    # Apply the mask to the original image to remove the right side of the line
    cropped_image_right = cv2.bitwise_and(image, image, mask=mask_right)

    # Convert the results to RGB for displaying with matplotlib
    cropped_image_left_rgb = cv2.cvtColor(cropped_image_left, cv2.COLOR_BGR2RGB)
    cropped_image_right_rgb = cv2.cvtColor(cropped_image_right, cv2.COLOR_BGR2RGB)

    # Plot the original image and the cropped images
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.title('Original')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.title('Remove Left Side')
    # plt.imshow(cropped_image_left_rgb)
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title('Remove Right Side')
    # plt.imshow(cropped_image_right_rgb)
    # plt.axis('off')

    # plt.show()

    # do bit wise and between the cropped_image_left_rgb and cropped_image_right_rgb
    # Apply the mask to the original image to remove the right side of the line
    cropped_image = cv2.bitwise_and(cropped_image_left_rgb, cropped_image_right_rgb)
    # plt.figure(figsize=(18, 6))
    # plt.title('Remove Both Sides')
    # plt.imshow(cropped_image)
    # plt.axis('off')
    # plt.show()

    """**APPLYING K-MEANS CLUSTERING**"""

    # Example usage
    takeYellow = True
    # current_temp = currentWeather['temperature'] # current temperature in celcious
    # num_clusters = adjust_clusters_based_on_temperature(current_temp)
    # print(f"Adjusted number of clusters: {num_clusters}")

    # Define criteria and apply K-means
    pixel_values = cropped_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    k = 8
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Mapping labels to center color
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(cropped_image.shape)

    # Visualize the clustered heatmap
    # plt.figure(figsize=(10, 5))
    # plt.title('Clustered Heat Map using K-means on RGB Values')
    # plt.imshow(segmented_image)
    # plt.colorbar()
    # plt.show()

    red_rgb = [255, 0, 0]
    yellow_rgb = [255, 255, 0]
    orange_rgb = [255, 165, 0]

    # Calculate the distance to identify the clusters for red and yellow
    distances_to_red = np.linalg.norm(centers - red_rgb, axis=1)
    distances_to_yellow = np.linalg.norm(centers - yellow_rgb, axis=1)
    distances_to_orange = np.linalg.norm(centers - orange_rgb, axis=1)

    red_cluster_index = np.argmin(distances_to_red)
    yellow_cluster_index = np.argmin(distances_to_yellow)
    orange_cluster_index = np.argmin(distances_to_orange)

    # print(_)
    # Create masks for red and yellow clusters
    red_mask = (labels == red_cluster_index).reshape(segmented_image.shape[:2])
    yellow_mask = (labels == yellow_cluster_index).reshape(segmented_image.shape[:2])
    orange_mask = (labels == orange_cluster_index).reshape(segmented_image.shape[:2])

    # Calculate the area by counting the number of pixels in each mask
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    orange_area = np.sum(orange_mask)

    if takeYellow:
        rotated_wanted_region = yellow_mask + red_mask + orange_mask
    else:
        rotated_wanted_region = red_mask + orange_mask

    # plt.title('Wanted Region Mask')
    # plt.imshow(wanted_region, cmap='gray')
    # plt.show()

    blurredForBlended, edgesForBlended = canny_edge_detection(blended_image[:170, :])
    # Display the original image and the detected edges
    # plt.figure(figsize=(15, 10))
    # plt.subplot(121)
    # plt.imshow(blurredForBlended, cmap='gray')
    # plt.title('Blurred Image')
    # plt.axis('off')

    # plt.subplot(122)
    # plt.imshow(edgesForBlended, cmap='gray')
    # plt.title('Canny Edge Detection')
    # plt.axis('off')
    # plt.show()

    # Function to detect lines and filter based on length

    # Set the minimum length for the lines
    min_length = 100  # Adjust this value based on your requirement

    # Detect long lines
    long_lines = detect_long_lines(edgesForBlended, min_length, max_gap=20, threshold=50)

    # Optionally, visualize the result
    output_img = cv2.cvtColor(edgesForBlended, cv2.COLOR_GRAY2BGR)
    for line in long_lines:
        cv2.line(output_img, line[0], line[1], (0, 255, 0), 2)

    # Assume long_lines is the list of detected lines
    swapped_lines = []

    for line in long_lines:
        (x1, y1), (x2, y2) = line
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        swapped_lines.append(((x1, y1), (x2, y2)))

    # Print the swapped lines
    # for line in swapped_lines:
    #     print(f"Start: {line[0]}, End: {line[1]}")
    swapped_lines = sorted(swapped_lines, key=lambda x: x[0][0], reverse=False)
    # Display the sorted lines
    # for line in swapped_lines:
    #     print(f"Start: {line[0]}, End: {line[1]}")
    # Assume swapped_lines is the list of detected lines after swapping
    threshold_distance = 70  # Adjust this threshold based on your requirement

    filtered_lines = []
    for i in range(len(swapped_lines)):
        if i == 0:
            filtered_lines.append(swapped_lines[i])
        else:
            (x1_prev, y1_prev), _ = swapped_lines[i - 1]
            (x1_curr, y1_curr), _ = swapped_lines[i]
            distance = np.sqrt((x1_curr - x1_prev)**2 + (y1_curr - y1_prev)**2)
            if distance > threshold_distance:
                filtered_lines.append(swapped_lines[i])

    # Print the filtered lines
    # for line in filtered_lines:
        # print(f"Start: {line[0]}, End: {line[1]}")

    # Assume filtered_lines is the list of lines after all previous processing
    # Get the middle x-coordinate of the image
    image_middle_x = edges.shape[1] // 2

    # Initialize variables to store the closest lines
    closest_left_lineForBlended = None
    closest_right_lineForBlended = None

    # Initialize minimum distances
    min_left_distance = float('inf')
    min_right_distance = float('inf')

    # Iterate through the filtered lines
    for line in filtered_lines:
        (x1, y1), (x2, y2) = line
        # Calculate the distance from the middle
        if x1 < image_middle_x:
            distance = abs(image_middle_x - x1)
            if distance < min_left_distance:
                min_left_distance = distance
                closest_left_lineForBlended = line
        elif x1 > image_middle_x:
            distance = abs(image_middle_x - x1)
            if distance < min_right_distance:
                min_right_distance = distance
                closest_right_lineForBlended = line

    # Return the closest lines
    # print(closest_left_lineForBlended)
    # print(closest_right_lineForBlended)

    # Load the image
    image = blended_image.copy()
    # Create a mask to remove the left side
    if closest_right_lineForBlended is not None:
        mask_left = create_mask_left(image.shape, closest_right_lineForBlended[0], closest_right_lineForBlended[1])

    # Create a mask to remove the right side
    if closest_left_lineForBlended is not None:
        mask_right = create_mask_right(image.shape, closest_left_lineForBlended[0], closest_left_lineForBlended[1])

    # Apply the mask to the original image to remove the left side of the line
    cropped_image_left = cv2.bitwise_and(image, image, mask=mask_left)

    # Apply the mask to the original image to remove the right side of the line
    cropped_image_right = cv2.bitwise_and(image, image, mask=mask_right)

    # Convert the results to RGB for displaying with matplotlib
    cropped_image_left_rgb = cv2.cvtColor(cropped_image_left, cv2.COLOR_BGR2RGB)
    cropped_image_right_rgb = cv2.cvtColor(cropped_image_right, cv2.COLOR_BGR2RGB)

    # Plot the original image and the cropped images
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.title('Original')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.title('Remove Left Side')
    # plt.imshow(cropped_image_left_rgb)
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title('Remove Right Side')
    # plt.imshow(cropped_image_right_rgb)
    # plt.axis('off')

    plt.show()
    # do bit wise and between the cropped_image_left_rgb and cropped_image_right_rgb
    # Apply the mask to the original image to remove the right side of the line
    cropped_image = cv2.bitwise_and(cropped_image_left_rgb, cropped_image_right_rgb)
    # plt.figure(figsize=(18, 6))
    # plt.title('Remove Both Sides')
    # plt.imshow(cropped_image)
    # plt.axis('off')
    # plt.show()

    croppedBlendedImage = cropped_image.copy()

    """Regarding dynamic thresholding, we want modify the number of clusters (k) based on the given time index and average temperature in the whole image, we can increase the number of clusters.
    
    Or, we can to determine the 'k' by creating a color histogram of the heat map
    to then determine how many dominant colors are in the image.
    
    If high road heat (average of entire image is high), we will increase the number of clusters and work only with the red/orange ranges of the image
    
    **DETERMINING THE NUMBER OF CLUSTERS**
    """

    pixel_values = cropped_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    k = 8
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Mapping labels to center color
    segmented_imageForBlended = centers[labels.flatten()]
    segmented_imageForBlended = segmented_imageForBlended.reshape(cropped_image.shape)

    # Visualize the clustered heatmap
    # plt.figure(figsize=(10, 5))
    # plt.title('Clustered Heat Map using K-means on RGB Values')
    # plt.imshow(segmented_imageForBlended)
    # plt.colorbar()
    # plt.show()

    red_rgb = [255, 0, 0]
    yellow_rgb = [255, 255, 0]
    orange_rgb = [255, 165, 0]

    # Calculate the distance to identify the clusters for red and yellow
    distances_to_red = np.linalg.norm(centers - red_rgb, axis=1)
    distances_to_yellow = np.linalg.norm(centers - yellow_rgb, axis=1)
    distances_to_orange = np.linalg.norm(centers - orange_rgb, axis=1)

    red_cluster_index = np.argmin(distances_to_red)
    yellow_cluster_index = np.argmin(distances_to_yellow)
    orange_cluster_index = np.argmin(distances_to_orange)

    # print(_)
    # Create masks for red and yellow clusters
    red_mask = (labels == red_cluster_index).reshape(segmented_imageForBlended.shape[:2])
    yellow_mask = (labels == yellow_cluster_index).reshape(segmented_imageForBlended.shape[:2])
    orange_mask = (labels == orange_cluster_index).reshape(segmented_imageForBlended.shape[:2])

    # Calculate the area by counting the number of pixels in each mask
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    orange_area = np.sum(orange_mask)

    # Visualize the masks
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Red Region Mask')
    # plt.imshow(red_mask, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Yellow Region Mask')
    # plt.imshow(yellow_mask, cmap='gray')
    # plt.show()
    #
    #
    # plt.subplot(2, 2, 2)
    # plt.title('Orange Region Mask')
    # plt.imshow(orange_mask, cmap='gray')
    # plt.show()


    # clear_output()

    if takeYellow:
        # print("Took yellow as part of ROI")
        wanted_regionForBlended = yellow_mask + red_mask + orange_mask
    else:
        # print("Did not take yellow as part of ROI")
        wanted_regionForBlended = red_mask + orange_mask

    #
    # plt.title('Wanted Region Mask')
    # plt.imshow(wanted_regionForBlended, cmap='gray')
    # plt.show()

    original2d = wanted_regionForBlended
    original2d = wanted_regionForBlended
    original_wanted_region_sum = np.sum(wanted_regionForBlended)

    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///

    # points_original = np.array([
    #     [closest_left_line[1][0],  segmented_image.shape[1]],
    #     [closest_right_line[1][0], segmented_image.shape[1]],
    #     [closest_right_line[0][0],  segmented_image.shape[0]],
    #     [closest_left_line[0][0], segmented_image.shape[0]]
    # ], dtype=np.float32)

    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///

    # points_topdown = np.array([
    #     [long_lines[0][0][0], 0],
    #     [long_lines[0][1][0], 0],
    #     [long_lines[1][1][0], segmented_image.shape[0]],
    #     [long_lines[1][0][0], segmented_image.shape[0]]
    # ], dtype=np.float32)

    # # /// === /// === /// === /// === /// === /// === /// === ///
    # # /// === /// === /// === /// === /// === /// === /// === ///
    # # /// === /// === /// === /// === /// === /// === /// === ///

    points_original = np.array([
        [closest_left_line[1][0], 0],
        [closest_right_line[1][0], 0],
        [closest_right_line[0][0], segmented_image.shape[0]],
        [closest_left_line[0][0], segmented_image.shape[0]]],
          dtype=np.float32)

    # # /// === /// === /// === /// === /// === /// === /// === ///
    # # /// === /// === /// === /// === /// === /// === /// === ///
    # # /// === /// === /// === /// === /// === /// === /// === ///

    points_topdown = np.array([
        [closest_left_lineForBlended[1][0], 0],
        [closest_right_lineForBlended[1][0], 0],
        [closest_right_lineForBlended[0][0], segmented_imageForBlended.shape[0]],
        [closest_left_lineForBlended[0][0], segmented_imageForBlended.shape[0]]
    ], dtype=np.float32)

    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///
    # /// === /// === /// === /// === /// === /// === /// === ///

    # Compute the homography matrix
    H, status = cv2.findHomography(points_original, points_topdown)

    # Ensure wanted_region is a numpy array and convert its data type
    wanted_region = np.array(rotated_wanted_region, dtype=np.uint8)

    # Apply the homography transformation to the wanted region mask
    height, width = wanted_region.shape
    warped_mask = cv2.warpPerspective(wanted_region, H, (width, height))

    # Calculate the area
    area_pixels = np.count_nonzero(warped_mask)

    # Display the warped mask
    # plt.title('Warped Wanted Region Mask')
    # plt.imshow(warped_mask, cmap='gray')
    # plt.show()

    # print(f"Size of warped mask: {warped_mask.shape}")
    # print(f"Size of wanted region: {wanted_region.shape}")

    # print(closest_right_lineForBlended)
    # print(closest_left_lineForBlended)

    """**CAULCATING AREA BY FINDING PIXELS TO METERS RATIO**
    
    Total pixels using np.sum(mask)
    """

    width = abs(closest_right_lineForBlended[0][0] - closest_left_lineForBlended[0][0])
    height = abs(closest_right_lineForBlended[0][1] - closest_right_lineForBlended[1][1])

    warpedWidth = abs(closest_right_line[0][0] - closest_left_line[0][0])
    warpedHeight = abs(closest_right_line[0][1] - closest_right_line[1][1])

    # print(f"Height: {height} pixels")

    roadLengthIRL = 100
    roadWidthIRL = 12

    ratioWith = roadWidthIRL / width
    ratioLength = roadLengthIRL / height

    warpedRatioWidth = roadWidthIRL / warpedWidth
    warpedRatioLength = roadLengthIRL / warpedHeight

    original_real_world_area = original_wanted_region_sum * ratioLength * ratioWith
    estimated_area_in_meters_pixel_conversion = area_pixels * warpedRatioLength * warpedRatioWidth

    # print("""************************RATIO CONVERSION*****************************""")

    # print("{:<40} {:>20}".format("Estimated area in meters:",
    #                              f"{estimated_area_in_meters_pixel_conversion:.2f} square meters"))
    # print("{:<40} {:>20}".format("Actual area in meters:", f"{original_real_world_area:.2f} square meters"))
    # print("{:<40} {:>20}".format("Estimated area in pixels:", f"{area_pixels}"))
    # print("{:<40} {:>20}".format("Actual area in pixels:", f"{original_wanted_region_sum}"))
    # print("""**************************CONTOURS************************************""")

    estimatedContours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimatedContours = max(estimatedContours, key=cv2.contourArea)

    actualContours, _ = cv2.findContours(wanted_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    actualContours = max(actualContours, key=cv2.contourArea)

    x_est, y_est, w_est, h_est = cv2.boundingRect(estimatedContours)
    x_act, y_act, w_act, h_act = cv2.boundingRect(actualContours)

    # Calculate areas of the bounding rectangles in pixels
    estimatedArea = w_est * h_est
    actualArea = w_act * h_act

    # Convert dimensions to real-world measurements
    # Assuming ratioLength and ratioWith are conversion factors from pixels to meters
    estimated_width_in_meters = w_est * ratioWith
    estimated_height_in_meters = h_est * ratioLength
    actual_width_in_meters = w_act * ratioWith
    actual_height_in_meters = h_act * ratioLength

    # Calculate real-world areas
    estimated_area_in_meters = estimated_width_in_meters * estimated_height_in_meters
    actual_area_in_meters = actual_width_in_meters * actual_height_in_meters

    # print("{:<40} {:>20}".format("Estimated area in meters:", f"{estimated_area_in_meters:.2f} square meters"))
    # print("{:<40} {:>20}".format("Actual area in meters:", f"{actual_area_in_meters:.2f} square meters"))
    # print("{:<40} {:>20}".format("Estimated area (bounding box):", f"{estimatedArea} pixels"))
    # print("{:<40} {:>20}".format("Actual area (bounding box):", f"{actualArea} pixels"))

    estimatedNoCars = estimate_cars(estimated_area_in_meters)
    estimatedNoCarsPixelToMeters = estimate_cars(estimated_area_in_meters_pixel_conversion)
    # print("************************NUMBER OF CARS********************************")
    # print("{:<40} {:>20}".format("Actual number of cars:", f"{estimate_cars(actual_area_in_meters)}"))
    # print("{:<40} {:>20}".format("Estimated number of cars using contours:", f"{estimatedNoCars}"))
    # print("{:<40} {:>20}".format("Estimated number of cars using pixel conversion:",
                                #  f"{estimatedNoCarsPixelToMeters}"))
    currentTimeGMT = datetime.datetime.now()
    currentTimeGMT = currentTimeGMT.replace(second=0, microsecond=0)
    formattedTimeGMT = currentTimeGMT.strftime("%H:%M")

    currentDate = datetime.datetime.now()
    formattedDate = currentDate.strftime("%m/%d/%Y")
    return estimatedNoCarsPixelToMeters,  currentWeather, formattedTimeGMT, formattedDate


def generate_estimation():
    history = []
    number_of_roads = 4  # intersection
    inputPatterns = [1, 2, 2, 3]  # 2 roads with same pattern (2)
    # generate random number between 0 to 3 exclusive
    random_number = np.random.randint(0, 3)
    # for i in range(number_of_roads):
    # trafficID = i+1
    estimation, weatherStamp, timeStamp, dateStamp = main(1, int(inputPatterns[random_number]))
    history.append([1, int(estimation), weatherStamp, timeStamp, dateStamp])
        # i += 1

    return history, int(estimation)


if __name__ == "__main__":
    estimations = generate_estimation()
    save_to_database(estimations)
    # print(f"Appended {estimations} to the database.")
    # print(f"Appended {estimations} to the database.")
