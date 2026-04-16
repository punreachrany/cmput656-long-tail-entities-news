import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Read the data directly from the CSV file
df = pd.read_csv('entity_label_counts.csv')

# 2. Drop the "Total" row so it doesn't break the chart's scale
df = df[df['Entity Label'] != 'Total']

# 3. Set up the figure and axes
fig, ax = plt.subplots(figsize=(16, 8))

# Define the x locations for the groups
x = np.arange(len(df))
width = 0.4  # the width of the bars

# 4. Plot the bars
# Shift the first set of bars to the left by half the width
ax.bar(x - width/2, df['Count'], width, label='Total Mentions (Count)', color='#4f9cf9')
# Shift the second set of bars to the right by half the width
ax.bar(x + width/2, df['Unique Count'], width, label='Unique Entities', color='#34d399')

# 5. Customizing the plot
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Entity Distribution: Total Mentions vs Unique Entities (Signal-1M Dataset)', fontsize=16, fontweight='bold', pad=15)

# Set the x-ticks to the middle of the two bars and label them
ax.set_xticks(x)
ax.set_xticklabels(df['Entity Label'], rotation=45, ha='right', fontsize=11)

# Apply Logarithmic scale for Y-axis because of the massive variance in counts
ax.set_yscale('log')

# Add legend and grid
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout so labels don't get cut off
plt.tight_layout()

# 6. Save and/or show
plt.savefig('entity_distribution_plot.png', dpi=300)
print("Plot saved as entity_distribution_plot.png")
plt.show()