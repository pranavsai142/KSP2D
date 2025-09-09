import numpy as np
import pygame
from ui_manager import UIManager
from geometry import Geometry, Circle, Foil
from environment import Air
from renderer import Renderer

# HYDROFOILER
# Main Menu Options
# The Shop
# Tunnel
# Ocean
# Pilot

# New driver


# End New driver

# Construct the object
foil = Foil("0012", 0.203, 300)
foil.geometry.plotGeometry("naca12_foil_geometry.png")
foil.geometry.plotNormals("naca12_foil_normals.png")

# circle = Circle(1, 200)
# circle.geometry.plotGeometry("circle_geometry.png")
# circle.geometry.plotNormals("circle_normals.png")

# Constants for flow generation
T = 100
OMEGA = 2 * np.pi / T
# OMEGA = 0
WIND_SPEED = 10
ALPHA = np.radians(0)
U_0 = WIND_SPEED  # Assuming U_0 should be WIND_SPEED based on context


# Initialize the Ocean
airEnvironment = Air()
airEnvironmentObject = airEnvironment.addObject(foil)  # Assuming 'foil' is defined

# Initialize the Renderer for Ocean
renderer = Renderer(windowWidth=1000, windowHeight=800)
uiManager = UIManager(1000, 800)

# Evolve the motion of the object
# leftKeyPressed = True
while airEnvironment.running:
#     if leftKeyPressed:
#         environmentObject.rotateLeft()
    airEnvironment.advanceTime()
    renderer.renderEnvironment(airEnvironment)
    uiManager.renderEnvironmentUI(airEnvironment, "ocean")
    pygame.display.flip()
#     ocean.printEnvironment()

renderer.quit()




