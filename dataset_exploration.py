import matplotlib.pyplot as plt

from utils.data.cornell_data import CornellDataset
from utils.visualisation import plot
import matplotlib.pyplot as plt
import cv2
import scipy.misc

if __name__ == "__main__":
    # experiement 1 loading the cornell data set
    cornell_data = CornellDataset('./cornell-dataset/')

    # Show the RGB Image
    image = cornell_data.get_rgb_img(83)
    grasps = cornell_data.get_gtbb(83)

    print("something")
    for gr in grasps:
        print(gr)
    fig, ax = plt.subplots()
    image.show(ax)
    grasps.show(ax)
    plt.show()

    # show the Depth Image
    image = cornell_data.get_depth_img(0)
    grasps = cornell_data.get_gtbb(0)

    fig, ax = plt.subplots()
    image.show(ax)
    grasps.show(ax)
    plt.show()
