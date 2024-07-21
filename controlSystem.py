from main import generate_estimation
from math import ceil
import time
import json
from datetime import datetime
# _, numOfCars = generate_estimation()
# print(_)
# numOfCars
# greenTime = min(numOfCars * 2 + 3, 120)
# print(numOfCars, greenTime)


class TrafficLight:
    def __init__(self, trafficID, lanes):
        self.trafficID = trafficID
        self.lanes = lanes
        self.totalCars = 0
        self.carsPerLane = 0
        self.greenTime = 0

    def update_traffic(self, totalCars):
        self.totalCars = totalCars
        self.carsPerLane = ceil(totalCars / self.lanes)
        # self.greenTime = min(self.carsPerLane * 2 + 3, 50)
        avgTimePerCar = 2.5
        self.greenTime = max(min(ceil((self.totalCars * avgTimePerCar) / (self.lanes + 1)), 50), 5)

    def getGreenTime(self):
        return self.greenTime

    def __str__(self):
        return (f"Traffic Light {self.trafficID} -> Lanes: {self.lanes}, Total Cars: {self.totalCars}, "
                f"Cars per Lane: {self.carsPerLane}, Green Time: {self.greenTime}s")
    # totalCars = 20  # Simulated number of cars


class CircularTrafficLights:
    def __init__(self, lights):
        self.lights = lights
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.lights:
            raise StopIteration
        light = self.lights[self.index]
        self.index = (self.index + 1) % len(self.lights)
        return light

    def update_light(self, trafficID, totalCars, carsPerLane, greenTime):
        for light in self.lights:
            if light.trafficID == trafficID:
                light.update_traffic(totalCars, carsPerLane, greenTime)
                print(f"Updated Traffic Light {trafficID}")


def save_to_database(data, filename='traffic_data.json'):
    try:
        with open(filename, 'r') as json_file:
            existing_data = json.load(json_file)
            if not isinstance(existing_data, list):
                existing_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    data_dict = {
        f"Cycle {data[0]}": {
            "trafficID": data[1],
            "num_cars": data[2],
            "weatherStamp:": data[3],
            "timeStamp": data[4],
            "dateStamp": data[5],
            "greenTime": data[6]
        }
    }

    existing_data.append(data_dict)  # Use append to add the dictionary to the list

    with open(filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


# Example usage:
lights = [
    TrafficLight(1, 9), # trafficID, lanes
    TrafficLight(2, 9),
    TrafficLight(3, 9),
    TrafficLight(4, 9)
]

traffic_circle = CircularTrafficLights(lights)

# Simulating updates and looping through them
i = 0
cycle = 0
for light in traffic_circle:
    if(i % len(lights) == 0):
        cycle += 1
    if(cycle == 10):
        break
    i += 1
    time.sleep(2)
    startTime = time.time()
    history, numOfCars = generate_estimation()
    endTime = time.time()
    light.update_traffic(numOfCars)
    data = [cycle, light.trafficID, numOfCars, history[0][2], history[0][3], history[0][4], light.getGreenTime()]
    save_to_database(data)
    print(light)
    print(f"Processing Time: {endTime-startTime}")
    time.sleep(light.getGreenTime())


filename = 'traffic_data.json'

# Open the file and load the data
with open(filename, 'r') as file:
    data = json.load(file)

# Now 'data' is a Python dictionary that contains the data from the JSON file
print(data)
