def read_data(file_path="data/nzt5/domain_app1.csv"):
    """Read data from file and convert to float values."""
    try:
        with open(file_path, "r") as f:
            text = f.read()
        # Convert all lines to floats, skipping any that can't be converted
        time_list = []
        for line in text.splitlines():
            try:
                time_list.append(float(line))
            except ValueError:
                continue
        return time_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

def calculate_average(file_path="data/nzt5/domain_app1.csv"):
    """Calculate the average of all values in the file."""
    data = read_data(file_path)
    if not data:
        return 0
    return sum(data) / len(data)

def calculate_average_above_one(file_path="data/nzt5/domain_app1.csv"):
    """Calculate the average of values > 1.0."""
    data = read_data(file_path)
    filtered_data = [value for value in data if value < 1.0]
    if not filtered_data:
        return 0
    return sum(filtered_data) / len(filtered_data)

def find_maximum(file_path="data/nzt5/domain_app1.csv"):
    """Find the maximum value in the file."""
    data = read_data(file_path)
    if not data:
        return 0
    return max(data)

def find_maximum_under_one(file_path="data/nzt5/domain_app1.csv"):
    """Find the maximum value that is < 1.0."""
    data = read_data(file_path)
    filtered_data = [value for value in data if value < 0.10]
    if not filtered_data:
        return 0
    return max(filtered_data)

def count_values_under_one(file_path="data/nzt5/domain_app1.csv"):
    """Count how many values are < 1.0."""
    data = read_data(file_path)
    return sum(1 for value in data if value < 1.0)

# Run the functions and display results
if __name__ == "__main__":
    file_path = "data/zt3/domain_app1.csv"
    print(f"Average of all values: {calculate_average(file_path)}")
    print(f"Average of values < 1.0: {calculate_average_above_one(file_path)}")
    print(f"Maximum value: {find_maximum(file_path)}")
    print(f"Maximum value < 1.0: {find_maximum_under_one(file_path)}")
    print(f"Count of values < 1.0: {count_values_under_one(file_path)}")