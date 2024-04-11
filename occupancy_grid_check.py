import numpy as np
import matplotlib.pyplot as plt

class OccupancyGrid:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.float32)  # Initialize with zeros (empty grid)

    def set_obstacle(self, x, y):
        # Set the cell corresponding to (x, y) as occupied
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        self.grid[grid_y, grid_x] = 1

    def visualize(self):
        plt.imshow(self.grid, cmap='binary', origin='lower')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.title('Occupancy Grid')
        plt.colorbar(label='Occupancy Probability')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define grid parameters
    width = 100  # Number of cells along the x-axis
    height = 100  # Number of cells along the y-axis
    resolution = 0.1  # Size of each cell in meters

    # Create an occupancy grid
    occupancy_grid = OccupancyGrid(width, height, resolution)

    # Set some obstacle cells
    occupancy_grid.set_obstacle(20, 30)
    occupancy_grid.set_obstacle(40, 50)
    occupancy_grid.set_obstacle(60, 70)

    # Visualize the occupancy grid
    occupancy_grid.visualize()
