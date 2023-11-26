import matplotlib.pyplot as plt
import numpy as np

# x = [43.82, 28.95, 43.48, 25.97, 66.43, 52.15, 105.28, 14.42, 14.72, 26.98]
# y = [80.6, 81.54, 80.55, 80.91, 81.48, 81.85, 82.07, 82.48, 82.65, 83.22]
x = [43.82, 28.95, 43.48, 25.97, 66.43, 52.15, 105.28, 14.42, 14.72, 26.98]
y = [70.71, 75.41, 76.08, 74.07, 63.53, 71.05, 76.74, 69.65, 72.42, 79.47]

plt.figure(figsize=(10, 30), dpi=90)
size = list()
color = list()
map = ['#EE7C68', '#FCDAD5', '#99FF99', '#99CCFF', '#FFCCFF', '#FFFCAD',
            '#c0c0c0', '#5f9e90', '#4682b4', '#ff7f00']
for c in map:
    color.append(c)
for tur in x:
    size.append(tur*35)

plt.xticks(range(10, 130, 10))
plt.yticks(range(60, 85, 5))
plt.scatter(x, y, c=color, s=size)
# plt.plot(x, y, color='blue')
plt.show()

# import matplotlib.pyplot as plt
# x = [1,2,3,4,5]
# y1 = [1,4,9,16,25]
# y2 = [2,4,8,18,30]
# plt.plot(x, y1, color='red', label='Parameter')
# plt.plot(x, y2, color='blue', label='DICE')
# plt.legend()
# plt.show()