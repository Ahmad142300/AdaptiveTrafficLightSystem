import json
import os


def read_counter(filename='counter.txt'):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write('0')
        return 0
    else:
        with open(filename, 'r') as file:
            return int(file.read().strip())


def update_counter(filename='counter.txt'):
    counter = read_counter(filename)
    counter += 1
    with open(filename, 'w') as file:
        file.write(str(counter))
    return counter


def save_to_database(data, filename='database.json'):
    try:
        with open(filename, 'r') as json_file:
            existing_data = json.load(json_file)
            if not isinstance(existing_data, list):
                existing_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    for entry in data:
        data_dict = {
            f"Cycle {read_counter()}": {
                "trafficID": entry[0],
                "num_cars": entry[1],
                "weatherStamp:": entry[2],
                "timeStamp": entry[3],
                "dateStamp": entry[4]
            }
        }

        existing_data.append(data_dict)  # Use append to add the dictionary to the list

    update_counter()

    with open(filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


