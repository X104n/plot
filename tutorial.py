import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Read the file and convert values to float
with open("data/domain_app.csv", "r") as f:
    text = f.read()

time_list = [float(x) for x in text.splitlines()]

# Create the plot
fig, ax = plt.subplots()
ax.plot(time_list)

# Set axis range
ax.axis((0, len(time_list), 0.0, 20.0))

# Format y-axis to show 2 decimal places
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Optional: Format x-axis too
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.show()
