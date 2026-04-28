import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ... plot your data ...
plt.show()  # drag to rotate, then check:
print(ax.elev, ax.azim)  # prints current angles
