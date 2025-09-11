import pygame
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import copy
from geometry import Geometry, Circle, Foil

MAX_FRAMES = 10000
PLOT_FRAMES = False
GRAVITY = 9.81
GRAVITY_VECTOR = np.array([0, -GRAVITY])
DELTA_T = 0.01
MAX_VELOCITY = 1000000
MAX_ACCELERATION = 1000000
THROTTLE_DELTA = 1.0
MAX_THROTTLE = 100
MIN_THROTTLE = -75

# NEGATIVE IS UP
MINIMUM_Z_COORDINATE = 0

TERRAIN_OBJECTS_MINIMUM_Z_COORDINATE = -1000
CLOUD_OBJECTS_MINIMUM_Z_COORDINATE = 100
CLOUD_OBJECTS_MAXIMUM_Z_COORDINATE = 5000


LAUNCHPAD_X_COORDINATE = -1

class Space:
    LAUNCHPAD_SEGMENT_WIDTH = 1
    LAUNCHPAD_SEGMENT_HEIGHT = 1
    
    def __init__(self, launchpadHeight=30):
        print("Initializing Space Environment")
        self.launchpadHeight = launchpadHeight
        self.objects = []
        self.frameNumber = 0
        self.frameFilenames = []
        self.orientationFrameFilenames = []
        self.objectFrameFilenames = []
        self.forceVectors = []
        self.positionVectors = []
        self.velocityVectors = []
        self.accelerationVectors = []
        self.addedMasses = []
        self.pathHistory = []  # For minimap
        self.running = True
        self.maxHistoryLength = 1000  # Cap history lengths
        self.deltaX = 1000
#         self.deltaZ = -MINIMUM_Z_COORDINATE * 2
        self.deltaZ = 10000
        self.terrainObjects = []
        self.cloudObjects = []
        self.launchpadObjects = []
        self.spawnTerrainObjects()
        self.spawnCloudObjects()
        self.spawnLaunchpadObjects()
    

    def addObject(self, geometryData):
        object = Object(geometryData, 0, 0)
        object.pointUp()
        self.objects.append(object)
        return object
        
    def spawnTerrainObjects(self):
        for _ in range(100):
            x = np.random.uniform(-self.deltaX/2, self.deltaX/2)
            z = np.random.uniform(TERRAIN_OBJECTS_MINIMUM_Z_COORDINATE, -25)
            radius = np.random.uniform(0.5, 50)
            self.terrainObjects.append({"pos": [x, z], "radius": radius})
            
    def spawnCloudObjects(self):
        for _ in range(100):
            x = np.random.uniform(-self.deltaX/2, self.deltaX/2)
            z = np.random.uniform(CLOUD_OBJECTS_MINIMUM_Z_COORDINATE, CLOUD_OBJECTS_MAXIMUM_Z_COORDINATE)
            vx = np.random.uniform(-2, 5)
            vz = np.random.uniform(-1, 1)
            radius = np.random.uniform(25, 100)
            self.cloudObjects.append({"pos": [x, z], "vel": [vx, vz], "radius": radius})
            
    def spawnLaunchpadObjects(self):
        x = LAUNCHPAD_X_COORDINATE
        zCoordinates = np.arange(0, self.launchpadHeight)
        for z in zCoordinates:
            self.launchpadObjects.append({"pos": [x, z]})
        

    def handleKey(self, event):
        if event.key == pygame.K_UP:
            for obj in self.objects:
                obj.throttleUp()
        elif event.key == pygame.K_DOWN:
            for obj in self.objects:
                obj.throttleDown()
        if event.key == pygame.K_LEFT:
            for obj in self.objects:
                obj.rotateLeft()
        elif event.key == pygame.K_RIGHT:
            for obj in self.objects:
                obj.rotateRight()
        elif event.key == pygame.K_k:
            for obj in self.objects:
                obj.killEngine()
                
    def handleKeys(self, keys):
        if keys[pygame.K_SPACE]:
            self.shootLaser()
        if keys[pygame.K_LEFT]:
            for obj in self.objects:
                obj.rotateLeft()
        if keys[pygame.K_RIGHT]:
            for obj in self.objects:
                obj.rotateRight()
        if keys[pygame.K_UP]:
            for obj in self.objects:
                obj.throttleUp()
        elif keys[pygame.K_DOWN]:
            for obj in self.objects:
                obj.throttleDown()
        elif keys[pygame.K_k]:
            for obj in self.objects:
                obj.killEngine()
        elif keys[pygame.K_r]:
            for obj in self.objects:
                obj.reverseEngine()
                
    def updateSize(self, deltaX, deltaZ):
        self.deltaX = max(10, deltaX)  # Minimum size to prevent issues
        self.deltaZ = max(10, deltaZ)
        
    def updateModifiableParameters(self, deltaX, deltaZ, enginePower, deltaRotation, rotationMin, rotationMax):
        self.updateSize(deltaX, deltaZ)
        for obj in self.objects:
            obj.enginePower = enginePower
            obj.deltaRotation = deltaRotation
            obj.rotationMin = np.radians(rotationMin)
            obj.rotationMax = np.radians(rotationMax)
                
    def advanceTime(self):
        for object in self.objects:
            object.updatePosition()
            self.pathHistory.append(object.positionVector.copy())
            if len(self.pathHistory) > self.maxHistoryLength:
                self.pathHistory.pop(0)
            if self.isOutsideBounds(object) or self.frameNumber >= MAX_FRAMES:
                print("CRASH!", object.positionVector, self.isOutsideBounds(object))
                if PLOT_FRAMES:
#                     self.createMovie("ocean_global_forces.gif", self.frameFilenames)
#                     self.createMovie("ocean_local_forces.gif", self.objectFrameFilenames)
#                     self.createMovie("ocean_local_orientations.gif", self.orientationFrameFilenames)
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_forces_timeseries.png"), self.forceVectors, "force")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_position_timeseries.png"), self.positionVectors, "position")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_velocity_timeseries.png"), self.velocityVectors, "velocity")
                    self.plotVectorTimeseries(os.path.join("graphs", "ocean_acceleration_timeseries.png"), self.accelerationVectors, "acceleration")
                    self.plotTimeseries(os.path.join("graphs", "ocean_added_mass_timeseries.png"), self.addedMasses, "added mass")
                self.running = False
            else:
                if PLOT_FRAMES:
#                     frameFilename = os.path.join("graphs", f"ocean_global_forces_{self.frameNumber}.png")
#                     orientationFrameFilename = os.path.join("graphs", f"ocean_local_orientation_{self.frameNumber}.png")
#                     self.plotForces(frameFilename, orientationFrameFilename, object)
#                     self.frameFilenames.append(frameFilename)
#                     self.orientationFrameFilenames.append(orientationFrameFilename)
#                     objectFrameFilename = os.path.join("graphs", f"ocean_local_forces_{self.frameNumber}.png")
#                     object.geometry.plotForces(objectFrameFilename, object.geometry.localVelocityVector,
#                                               object.geometry.tangentialTotalVelocity, object.geometry.localForceVector)
#                     self.objectFrameFilenames.append(objectFrameFilename)
#                     # Cap frame filenames
#                     if len(self.frameFilenames) > self.maxHistoryLength:
#                         self.frameFilenames.pop(0)
#                     if len(self.orientationFrameFilenames) > self.maxHistoryLength:
#                         self.orientationFrameFilenames.pop(0)
#                     if len(self.objectFrameFilenames) > self.maxHistoryLength:
#                         self.objectFrameFilenames.pop(0)
                    self.forceVectors.append(object.forceVector)
                    self.positionVectors.append(copy.copy(object.positionVector))
                    self.velocityVectors.append(copy.copy(object.velocityVector))
                    self.accelerationVectors.append(copy.copy(object.accelerationVector))
                    self.addedMasses.append(copy.copy(object.addedMass))
                    # Cap vector lists
                    if len(self.forceVectors) > self.maxHistoryLength:
                        self.forceVectors.pop(0)
                    if len(self.positionVectors) > self.maxHistoryLength:
                        self.positionVectors.pop(0)
                    if len(self.velocityVectors) > self.maxHistoryLength:
                        self.velocityVectors.pop(0)
                    if len(self.accelerationVectors) > self.maxHistoryLength:
                        self.accelerationVectors.pop(0)
                    if len(self.addedMasses) > self.maxHistoryLength:
                        self.addedMasses.pop(0)
                self.frameNumber += 1
        self.updateClouds()
                
#                 
    def updateClouds(self):
        for cloudObject in self.cloudObjects:
            cloudObject["pos"][0] += cloudObject["vel"][0] * DELTA_T
            cloudObject["pos"][1] += cloudObject["vel"][1] * DELTA_T
            if cloudObject["pos"][0] < -self.deltaX/2 or cloudObject["pos"][0] > self.deltaX/2:
                cloudObject["vel"][0] = -cloudObject["vel"][0]
            if cloudObject["pos"][1] < -self.deltaZ or cloudObject["pos"][1] > CLOUD_OBJECTS_MINIMUM_Z_COORDINATE:
                cloudObject["vel"][1] = -cloudObject["vel"][1]

                
    def cleanup(self):
        # Clear lists to free memory, but preserve objects (geometry)
        self.pathHistory.clear()
        self.frameFilenames.clear()
        self.orientationFrameFilenames.clear()
        self.objectFrameFilenames.clear()
        self.forceVectors.clear()
        self.positionVectors.clear()
        self.velocityVectors.clear()
        self.accelerationVectors.clear()
        self.addedMasses.clear()
        # Close any open Matplotlib figures
        plt.close('all')


    def isOutsideBounds(self, object):
        return False
#         return (object.positionVector[1] < MINIMIM_Z_COORDINATE)

    def plotForces(self, filename, orientationFilename, object):
        orient = np.array(object.orientationVector, dtype=float)
        force = np.array(object.forceVector, dtype=float)
        norm = np.sqrt(orient @ orient)
        local_x_axis = orient / norm
        local_z_axis = np.array([-local_x_axis[1], local_x_axis[0]])
        globalXCoords = []
        globalZCoords = []
        for x_local, z_local in zip(object.geometry.pointXCoords, object.geometry.pointZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis + object.positionVector
            globalXCoords.append(global_point[0])
            globalZCoords.append(global_point[1])
        globalColocationXCoords = []
        globalColocationZCoords = []
        for x_local, z_local in zip(object.geometry.colocationXCoords, object.geometry.colocationZCoords):
            global_point = x_local * (-local_x_axis) + z_local * local_z_axis + object.positionVector
            globalColocationXCoords.append(global_point[0])
            globalColocationZCoords.append(global_point[1])
        centroidX = np.mean(globalXCoords)
        centroidZ = np.mean(globalZCoords)
        apparentCurrentVector = -np.array(object.velocityVector)
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(max(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.arrow(centroidX, centroidZ,
                  object.forceVector[0] * 0.01, object.forceVector[1] * 0.01,
                  head_width=0.01, head_length=0.01, fc='pink', ec='pink', label='Force Vector')
#         plt.xlim([0, self.deltaX])
#         plt.ylim([-self.deltaZ, 1])
        plt.title("Global Velocity Frame")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        plt.grid(True)
        plt.axis('equal')
        plt.scatter(globalXCoords, globalZCoords, label="points", s=5)
        scale = 0.1
        plt.arrow(min(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red', label='Apparent Velocity')
        plt.arrow(max(globalXCoords), min(globalZCoords) - 0.1,
                  apparentCurrentVector[0] * scale/2, apparentCurrentVector[1] * scale/2,
                  head_width=0.01, head_length=0.01, fc='red', ec='red')
        plt.title("Orientation Frame")
        plt.legend()
        plt.savefig(orientationFilename)
        plt.close()

    def createMovie(self, movieFilename, frameFilenames):
        with imageio.get_writer(os.path.join('graphs', movieFilename), mode='I', duration=0.1) as writer:
            for frame in frameFilenames:
                image = imageio.imread(frame)
                writer.append_data(image)
                os.remove(frame)

    def plotVectorTimeseries(self, filename, vectors, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), [vector[0] for vector in vectors], label='X', linestyle='--')
        plt.plot(np.arange(self.frameNumber), [vector[1] for vector in vectors], label='Z')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(f"{title} vs time")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plotTimeseries(self, filename, values, title):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.frameNumber), values, label=title, linestyle='--')
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.title(f"{title} vs time")
        plt.legend()
        plt.savefig(filename)
        plt.close()

class Object:
    def __init__(self, geometryData, positionX, positionZ):
        self.geometryData = geometryData
        self.geometry = self.geometryData.geometry
        self.mass = geometryData.mass
        self.positionVector = np.array([positionX, positionZ], dtype=np.float64)
        self.velocityVector = np.array([0.0, 0.0], dtype=np.float64)
        self.accelerationVector = GRAVITY_VECTOR / self.mass
        self.orientationVector = np.array([0.0, 0.0], dtype=np.float64)
        self.forceVector = np.array([0.0, 0.0], dtype=np.float64)
        self.totalForceVector = np.array([0.0, 0.0], dtype=np.float64)
        self.thrustForce = np.array([0, 0], dtype=np.float64)
        self.engineThrottle = 0.0
        self.addedMass = 0.0
#         Modifiable fields
        self.enginePower = 5000
        self.deltaRotation = 0.1
        self.rotationMin = np.radians(-15)
        self.rotationMax = np.radians(15)

    def rotateRight(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = np.arctan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = np.radians(np.degrees(currentAngle) + self.deltaRotation)
            if(newAngle > self.rotationMin and newAngle < self.rotationMax):
                self.orientationVector = np.array([np.cos(newAngle), np.sin(newAngle)], dtype=np.float64)
                self.setThrustForce()

    def rotateLeft(self):
        if self.geometryData.hasTrailingEdge:
            currentAngle = np.arctan2(self.orientationVector[1], self.orientationVector[0])
            newAngle = np.radians(np.degrees(currentAngle) - self.deltaRotation)
            if(newAngle > self.rotationMin and newAngle < self.rotationMax):
                self.orientationVector = np.array([np.cos(newAngle), np.sin(newAngle)], dtype=np.float64)
                self.setThrustForce()
            
    def throttleUp(self):
        if(self.engineThrottle < MAX_THROTTLE):
            self.engineThrottle += THROTTLE_DELTA
            self.setThrustForce()

    def throttleDown(self):
        if(self.engineThrottle > 0):
            self.engineThrottle -= THROTTLE_DELTA
            self.setThrustForce()      

    def reverseEngine(self):
        if(self.engineThrottle > MIN_THROTTLE):
            self.engineThrottle -= THROTTLE_DELTA
            self.setThrustForce()      
    
    def setThrustForce(self):
        self.thrustForce = self.orientationVector * self.enginePower * (self.engineThrottle / 100.0)  # 1 m/s^2 acceleration
        
    def killEngine(self):
        self.engineThrottle = 0
        self.thrustForce = np.array([0, 0], dtype=np.float64)

    def pointDown(self):
        self.orientationVector = np.array([0.0, -1.0], dtype=np.float64)
        
    def pointLeft(self):
        self.orientationVector = np.array([-1.0, 0.0], dtype=np.float64)
        
    def pointRight(self):
        self.orientationVector = np.array([1.0, 0.0], dtype=np.float64)
        
    def pointUp(self):
        self.orientationVector = np.array([0.0, 1.0], dtype=np.float64)

    def capAcceleration(self):
        self.accelerationVector = np.clip(self.accelerationVector, -MAX_ACCELERATION, MAX_ACCELERATION)

    def capVelocity(self):
        self.velocityVector = np.clip(self.velocityVector, -MAX_VELOCITY, MAX_VELOCITY)

    def updatePosition(self):
        self.updateForce(self.velocityVector, self.accelerationVector)
#         print("self.forceVector", self.forceVector)
#         modifiedForceVector = [self.forceVector[0], self.forceVector[1] * -1]
#         totalForceVector = self.forceVector + ((self.mass + self.addedMass) * GRAVITY_VECTOR) + self.thrustForce
        self.totalForceVector = self.forceVector + ((self.mass) * GRAVITY_VECTOR) + self.thrustForce
#         Apply normal force if on the ground. (Zero z component of force)
#         if(self.positionVector[1] <= MINIMUM_Z_COORDINATE):
#             totalForceVector[1] = 0
#         TODO: Need to figure out how to represent the ground. So I can have a runway.
#         self.accelerationVector = totalForceVector / (self.mass + self.addedMass)
        self.accelerationVector = self.totalForceVector / (self.mass)
        self.capAcceleration()
        self.velocityVector += self.accelerationVector * DELTA_T
        self.capVelocity()
        newPositionVector = self.positionVector + (self.velocityVector * DELTA_T)
#         Apply logic to detect if new postion is still on ground.
#          If so, negate vertical velocity and set vertical position to ground.
        if(newPositionVector[1] <= MINIMUM_Z_COORDINATE):
            self.accelerationVector[1] = 0
            self.velocityVector[1] = 0
            self.positionVector[1] = MINIMUM_Z_COORDINATE
            self.positionVector[0] = newPositionVector[0]
        else:
            self.positionVector = newPositionVector

    def updateForce(self, velocityVector, accelerationVector):
        self.velocityVector = velocityVector
        self.accelerationVector = accelerationVector
        self.forceVector = self.geometry.computeForceFromFlow(self.orientationVector, velocityVector, accelerationVector)
        accelerationMagnitude = np.sqrt(self.accelerationVector[0]**2 + self.accelerationVector[1]**2)
        forceMagnitude = np.sqrt(self.forceVector[0]**2 + self.forceVector[1]**2)
        self.addedMass = forceMagnitude / accelerationMagnitude if accelerationMagnitude > 0 else 0.0