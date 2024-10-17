import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 50)
y = np.sin(x)
y_err = 0.2  # Standard error

# Create the line plot
plt.plot(x, y, color='blue')

# Add the standard error ribbon
plt.fill_between(x, y - y_err, y + y_err, color='blue', alpha=0.2)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot with Standard Error Ribbon')

# Show the plot
plt.show()