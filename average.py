with open("data/nzt5/domain_app5.csv", "r") as f:
    text = f.read()
time_list = text.splitlines()
i = 0
total = 0.0
while True:
    total += float(time_list[i])
    i += 1
    if i >= len(time_list):
        break


print(f"Average time: {total / i}")

def maximum():
    """
    Calculate the maximum value from a list of numbers in a file.
    """
    with open("data/nzt5/domain_app5.csv", "r") as f:
        text = f.read()
    time_list = text.splitlines()

    max_value = float(time_list[0])
    for time in time_list:
        if float(time) > max_value:
            max_value = float(time)

    print(f"Maximum value: {max_value}")

print(maximum())