with open("data/domain_app_overnight_zt.csv", "r") as f:
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