import pygame
import numpy as np
import math
import random

import Constants as const

class Renderer:
    ZOOM = 0.01
    TUNNEL_ZOOM = 0.01
    SHOW_GLOBAL_FORCE_VECTOR = True
    SPACE_ZOOM_FACTOR = 10000

    def __init__(self, windowWidth=1000, windowHeight=800):
        pygame.init()
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.screen = pygame.display.set_mode((windowWidth, windowHeight), pygame.RESIZABLE)
        pygame.display.set_caption("KSP(OS)")
        self.bgColor = (255, 255, 255)
        self.pointColor = (0, 0, 255)
        self.lineColor = (0, 0, 255)
        self.tangentialColor = (0, 255, 0)
        self.velocityColor = (255, 0, 0)
        self.forceColor = (255, 105, 180)
        self.thrustColor = (0, 0, 255)
        self.asteroidColor = (100, 100, 100)
        self.laserColor = (255, 0, 0)
        self.oceanColor = (0, 100, 255)
        self.groundColor = (139, 69, 19)
        self.moonColor = (115, 114, 111)
        self.marsColor = (218, 169, 123)
        self.boulderColor = (166, 129, 61)
        self.moonBoulderColor = (151, 150, 146)
        self.marsBoulderColor = (90, 77, 77)
        self.runwayColor = (17, 18, 20)
        self.cloudColor = (220, 220, 220)
        self.orientBgColor = (200, 200, 200, 128)
        self.orbitColor = (128, 128, 128)  # Gray for orbital trajectory
        self.periColor = (255, 0, 0)       # Red for periapsis
        self.apoColor = (0, 0, 255)       # Blue for apoapsis
        self.atmosphereColors = [
            (0, 0, 0),        # Top (space)
            (0, 0, 139),      # Dark blue
            (25, 25, 112),    # Midnight blue
            (70, 130, 180),   # Steel blue
            (100, 149, 237),  # Cornflower blue
            (135, 206, 235)   # Sky blue (ground)
        ]
        self.marsAtmosphereColors = [
            (0, 0, 0),        # Top (space)
            (132, 140, 139),  # Transition color
            (209, 191, 162)   # Low altitude (ground)
        ]
        self.moonAtmosphereColor = (0, 0, 0)
        self.environment = None
        self.environmentObj = None
        self.zoomFactor = 1.0
        self.globalOffsetX = 0.0
        self.globalOffsetZ = 0.0
        self.prevSubDomainCenter = [0, 0]
        self.deltaX = 100.0
        self.deltaZ = 100.0
        self._updateDimensions(windowWidth, windowHeight)
        self.running = True
        self.firstRender = True

    def _getAtmosphereColor(self, altitude, deltaZ):
        if self.environment == "moon":
            return self.moonAtmosphereColor
        if self.environment == "mars":
            if deltaZ <= 0:
                return self.marsAtmosphereColors[-1][:3]
            frac = max(0, min(1, altitude / deltaZ))
            frac = 1 - frac
            num_colors = len(self.marsAtmosphereColors)
            index = frac * (num_colors - 1)
            i = int(index)
            t = index - i
            if i >= num_colors - 1:
                return self.marsAtmosphereColors[-1][:3]
            c1 = self.marsAtmosphereColors[i]
            c2 = self.marsAtmosphereColors[i + 1]
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            return (r, g, b)
        if deltaZ <= 0:
            return self.atmosphereColors[-1][:3]
        frac = max(0, min(1, altitude / deltaZ))
        frac = 1 - frac
        num_colors = len(self.atmosphereColors)
        index = frac * (num_colors - 1)
        i = int(index)
        t = index - i
        if i >= num_colors - 1:
            return self.atmosphereColors[-1][:3]
        c1 = self.atmosphereColors[i]
        c2 = self.atmosphereColors[i + 1]
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        return (r, g, b)

    def _computeGeometryZoom(self, environmentObj):
        maxSize = 0
        for obj in environmentObj.objects:
            xRange = max(obj.geometry.pointXCoords) - min(obj.geometry.pointXCoords)
            zRange = max(obj.geometry.pointZCoords) - min(obj.geometry.pointZCoords)
            maxSize = max(maxSize, xRange, zRange)
        if self.environment == "tunnel":
            maxSize = max(maxSize, 0.5)
        return maxSize * self.zoomFactor

    def _updateDimensions(self, windowWidth, windowHeight):
        # Store previous dimensions to adjust offsets
        prev_width = self.windowWidth
        prev_height = self.windowHeight
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        readoutWidth = windowWidth // 6
        self.globalWidth = windowWidth - readoutWidth
        self.globalHeight = windowHeight
        self.globalPos = (0, 0)
        self.orientHeight = int(windowHeight // 2.5)
        self.orientWidth = self.orientHeight
        self.orientPos = (10, windowHeight - self.orientHeight - 10)
        self.minimapWidth = windowWidth // 6
        self.minimapHeight = windowWidth // 6
        self.minimapPos = (0, 0)
        self.minimapScaleX = self.minimapWidth / self.deltaX if self.deltaX else 1.0
        self.minimapScaleZ = self.minimapHeight / self.deltaZ if self.deltaZ else 1.0
        # Adjust offsets to keep object centered
        if prev_width and prev_height and self.environmentObj:
            # Scale offsets based on window size change
            width_ratio = windowWidth / prev_width
            height_ratio = windowHeight / prev_height
            self.globalOffsetX *= width_ratio
            self.globalOffsetZ *= height_ratio
            self.updateViewParameters()

    def resize(self, size):
        self.windowWidth, self.windowHeight = size
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self._updateDimensions(*size)
        if self.environmentObj:
            self.prevSubDomainCenter = self._computeSubDomainCenter(self.environmentObj)
            self.updateViewParameters()

    def renderMenu(self, mainMenu):
        mainMenu.render(self.screen, geometrySet=self._checkGeometrySet())
        pygame.display.flip()

    def _checkGeometrySet(self):
        return True

    def renderShop(self, shop):
        shop.render(self.screen)
        pygame.display.flip()

    def handleEvent(self, event, ui_manager=None, ui_active=False):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Zoom in
                self.zoomFactor *= 0.95
            elif event.button == 5:  # Zoom out
                self.zoomFactor *= 1.05
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                self.resetView()
            elif event.key == pygame.K_d:
                self.globalOffsetX -= 0.5 / self.scaleX
            elif event.key == pygame.K_a:
                self.globalOffsetX += 0.5 / self.scaleX
            elif event.key == pygame.K_w:
                self.globalOffsetZ -= 0.5 / self.scaleZ
            elif event.key == pygame.K_s:
                self.globalOffsetZ += 0.5 / self.scaleZ
        if self.environmentObj:
            self.updateViewParameters()

    def resetView(self):
        if self.environment == "space":
            self.zoomFactor = self.SPACE_ZOOM_FACTOR
        else:
            self.zoomFactor = 1.0
        self.globalOffsetX = 0.0
        self.globalOffsetZ = 0.0
        if self.environmentObj:
            self.prevSubDomainCenter = self._computeSubDomainCenter(self.environmentObj)
            self.updateViewParameters()

    def renderEnvironment(self, environmentObj):
        self.environment = environmentObj.__class__.__name__.lower()
        self.environmentObj = environmentObj
        if self.environment in ["ocean", "pilot"]:
            self.deltaX = getattr(environmentObj, 'deltaX', 100.0)
            self.deltaZ = getattr(environmentObj, 'deltaZ', 100.0)
        elif self.environment in ["air", "space", "moon", "mars"]:
            self.deltaX = getattr(environmentObj, 'deltaX', 100.0)
            self.deltaZ = getattr(environmentObj, 'deltaZ', 100.0)
        else:
            self.deltaX = 100.0
            self.deltaZ = 100.0

        self.minimapScaleX = self.minimapWidth / self.deltaX if self.deltaX else 1.0
        self.minimapScaleZ = self.minimapHeight / self.deltaZ if self.deltaZ else 1.0
        self.prevSubDomainCenter = self._computeSubDomainCenter(environmentObj)
        if self.firstRender:
            if self.environment == "space":
                self.zoomFactor = self.SPACE_ZOOM_FACTOR
            else:
                self.zoomFactor = 1.0
            self.globalOffsetX = 0.0
            self.globalOffsetZ = 0.0
            self.firstRender = False
        self.updateViewParameters()
        if self.environment in ["air", "space", "moon", "mars"] and environmentObj.objects:
            if self.environment == "space":
                altitude = environmentObj.objects[0].latLonHeight[2]
                self.bgColor = self._getAtmosphereColor(altitude, const.VON_KARMAN_LINE)
            else:
                altitude = environmentObj.objects[0].positionVector[1]
                self.bgColor = self._getAtmosphereColor(altitude, self.deltaZ)
        else:
            self.bgColor = (255, 255, 255)
        self.screen.fill(self.bgColor)
        self._renderGlobalView(environmentObj, self.screen)
        self._renderOrientationView(environmentObj, self.screen)
        if self.environment in ["ocean", "pilot", "air", "space", "moon", "mars"]:
            self._renderMinimapView(environmentObj, self.screen)

    def updateViewParameters(self):
        if not self.environmentObj:
            return
        zoom = self._computeGeometryZoom(self.environmentObj)
        self.subDomainSizeX = max(1.2 * zoom, 0.05)
        self.subDomainSizeZ = max(1.2 * zoom, 0.05)
        if self.zoomFactor < 1e-6:
            self.zoomFactor = 1e-6
        self.scaleX = self.globalWidth / self.subDomainSizeX
        self.scaleZ = self.globalHeight / self.subDomainSizeZ
        # Update subdomain center to track object
        if self.environmentObj.objects:
            self.prevSubDomainCenter = self._computeSubDomainCenter(self.environmentObj)

    def _toScreenCoords(self, x, z, isGlobalView=False, isOrientView=False, isMinimapView=False, 
                        orientCenter=None, orientScale=None, subDomainCenter=None, isStatic=False):
        if isGlobalView:
            if isStatic:
                x = (x + self.globalOffsetX) * self.scaleX + self.globalPos[0] + self.globalWidth / 2
                z = -(z + self.globalOffsetZ) * self.scaleZ + self.globalPos[1] + self.globalHeight / 2
            else:
                if subDomainCenter is not None:
                    x = x - subDomainCenter[0]
                    z = z - subDomainCenter[1]
                x = (x + self.globalOffsetX) * self.scaleX + self.globalPos[0] + self.globalWidth / 2
                z = -(z + self.globalOffsetZ) * self.scaleZ + self.globalPos[1] + self.globalHeight / 2
        elif isOrientView:
            x = (x - orientCenter[0]) * orientScale + self.orientPos[0] + self.orientWidth / 2
            z = -(z - orientCenter[1]) * orientScale + self.orientPos[1] + self.orientHeight / 2
        elif isMinimapView:
            if self.environment in ["ocean", "pilot"]:
                x = max(0, min(self.deltaX, x))
                z = max(-self.deltaZ, min(0, z))
                screenX = (x / self.deltaX) * self.minimapWidth + self.minimapPos[0]
                screenZ = ((0 - z) / self.deltaZ) * self.minimapHeight + self.minimapPos[1]
            elif self.environment == "space":
                # Scale zoomed out to show entire orbit
                if self.environmentObj and self.environmentObj.objects:
                    a = getattr(self.environmentObj.objects[0], 'semimajor', self.environmentObj.earthRadius)
                    # Zoom out by increasing denominator (e.g., 3x max of earthRadius or semimajor)
                    scale = min(self.minimapWidth, self.minimapHeight) / (3 * max(self.environmentObj.earthRadius, a * (1 + getattr(self.environmentObj.objects[0], 'eccentricity', 0))))
                    screenX = x * scale + self.minimapPos[0] + self.minimapWidth / 2
                    screenZ = -z * scale + self.minimapPos[1] + self.minimapHeight / 2
                else:
                    screenX = self.minimapPos[0] + self.minimapWidth / 2
                    screenZ = self.minimapPos[1] + self.minimapHeight / 2
            else:  # air, moon, mars
                x = max(-self.deltaX/2, min(self.deltaX/2, x))
                z = max(0, min(self.deltaZ, z))
                screenX = ((x + self.deltaX/2) / self.deltaX) * self.minimapWidth + self.minimapPos[0]
                screenZ = ((self.deltaZ - z) / self.deltaZ) * self.minimapHeight + self.minimapPos[1]
            return (screenX, screenZ)
        return (x, z)

    def _drawArrow(self, surface, start, end, color, headSize=10):
        pygame.draw.line(surface, color, start, end, 2)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        dx, dy = dx / length, dy / length
        perpX, perpY = -dy, dx
        headPoint1 = (end[0] - headSize * dx + headSize * perpX / 2, end[1] - headSize * dy + headSize * perpY / 2)
        headPoint2 = (end[0] - headSize * dx - headSize * perpX / 2, end[1] - headSize * dy - headSize * perpY / 2)
        pygame.draw.polygon(surface, color, [end, headPoint1, headPoint2])

    def _computeSubDomainCenter(self, environmentObj):
        for obj in environmentObj.objects:
            return obj.positionVector
        return [0, 0]

    def _renderGlobalView(self, environmentObj, surface):
        subDomainCenter = self._computeSubDomainCenter(environmentObj)
        centerX, centerZ = subDomainCenter
        xMin = centerX - self.subDomainSizeX / 2
        xMax = centerX + self.subDomainSizeX / 2
        zMin = centerZ - self.subDomainSizeZ / 2
        zMax = centerZ + self.subDomainSizeZ / 2

        if self.environment in ["ocean", "pilot"]:
            x_bound_min, x_bound_max = 0, self.deltaX
            z_bound_min, z_bound_max = -self.deltaZ, 0
        elif self.environment in ["air", "space", "moon", "mars"]:
            x_bound_min, x_bound_max = -self.deltaX/2, self.deltaX/2
            z_bound_min, z_bound_max = 0, self.deltaZ
        else:  # tunnel
            x_bound_min, x_bound_max = -0.5, 0.5
            z_bound_min, z_bound_max = -0.5, 0.5

        if self.environment in ["ocean", "pilot"]:
            if xMin <= x_bound_max <= xMax:
                top = self._toScreenCoords(x_bound_max, max(zMin, z_bound_min), isGlobalView=True, subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(x_bound_max, min(zMax, z_bound_max), isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            if xMin <= x_bound_min <= xMax:
                top = self._toScreenCoords(x_bound_min, max(zMin, z_bound_min), isGlobalView=True, subDomainCenter=subDomainCenter)
                bottom = self._toScreenCoords(x_bound_min, min(zMax, z_bound_max), isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, top, bottom, 2)
            if zMin <= z_bound_min <= zMax:
                left = self._toScreenCoords(max(xMin, x_bound_min), z_bound_min, isGlobalView=True, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, x_bound_max), z_bound_min, isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.groundColor, left, right, 3)
            if zMin <= z_bound_max <= zMax:
                left = self._toScreenCoords(max(xMin, x_bound_min), z_bound_max, isGlobalView=True, subDomainCenter=subDomainCenter)
                right = self._toScreenCoords(min(xMax, x_bound_max), z_bound_max, isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.oceanColor, left, right, 2)

        if self.environment in ["air", "space", "moon", "mars"]:
            left = self._toScreenCoords(-self.deltaX, 0, isGlobalView=True, subDomainCenter=subDomainCenter)
            right = self._toScreenCoords(self.deltaX, 0, isGlobalView=True, subDomainCenter=subDomainCenter)
            ground_rect = pygame.Rect(left[0], left[1], right[0] - left[0], self.globalHeight - left[1])
            if self.environment == "air":
                pygame.draw.rect(surface, self.groundColor, ground_rect)
            elif self.environment == "moon":
                pygame.draw.rect(surface, self.moonColor, ground_rect)
            elif self.environment == "mars":
                pygame.draw.rect(surface, self.marsColor, ground_rect)
            elif self.environment == "space":
                earthCenter = self._toScreenCoords(0, 0, isGlobalView=True, subDomainCenter=subDomainCenter)
                clipTopLeft = self._toScreenCoords(xMin - self.globalOffsetX, zMin - self.globalOffsetZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                clipBottomRight = self._toScreenCoords(xMax - self.globalOffsetX, zMax - self.globalOffsetZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                clip = pygame.Rect((clipTopLeft[0], clipBottomRight[1], clipBottomRight[0] - clipTopLeft[0], clipTopLeft[1] - clipBottomRight[1]))
                surface.set_clip(clip)
                pygame.draw.circle(surface, self.groundColor, earthCenter, self.environmentObj.earthRadius * self.scaleX)
                surface.set_clip(None)
                # Orbital trajectory
#                 if environmentObj.objects:
#                     obj = environmentObj.objects[0]
#                     a = getattr(obj, 'semimajor', 0)
#                     e = getattr(obj, 'eccentricity', 0)
#                     nu = getattr(obj, 'trueanomaly', 0)
#                     pos = np.array(obj.positionVector)
#                     r = np.linalg.norm(pos)
#                     if r > 0 and a > 0:
#                         phi = math.atan2(pos[1], pos[0])
#                         omega = phi - nu
#                         num_points = 100
#                         trajectory = []
#                         for i in range(num_points + 1):
#                             theta = 2 * math.pi * i / num_points
#                             r_theta = a * (1 - e**2) / (1 + e * math.cos(theta))
#                             tx = r_theta * math.cos(theta + omega)
#                             tz = r_theta * math.sin(theta + omega)
#                             trajectory.append((tx, tz))
#                         prev_screen = None
#                         for px, pz in trajectory:
#                             screen_pos = self._toScreenCoords(px, pz, isGlobalView=True, subDomainCenter=subDomainCenter)
#                             if prev_screen is not None:
#                                 pygame.draw.line(surface, self.orbitColor, prev_screen, screen_pos, 1)
#                             prev_screen = screen_pos
#                         # Periapsis point
#                         r_peri = a * (1 - e)
#                         theta_peri = 0
#                         px_peri = r_peri * math.cos(theta_peri + omega)
#                         pz_peri = r_peri * math.sin(theta_peri + omega)
#                         screen_peri = self._toScreenCoords(px_peri, pz_peri, isGlobalView=True, subDomainCenter=subDomainCenter)
#                         pygame.draw.circle(surface, self.periColor, screen_peri, 8)
#                         # Apoapsis point
#                         r_apo = a * (1 + e)
#                         theta_apo = math.pi
#                         px_apo = r_apo * math.cos(theta_apo + omega)
#                         pz_apo = r_apo * math.sin(theta_apo + omega)
#                         screen_apo = self._toScreenCoords(px_apo, pz_apo, isGlobalView=True, subDomainCenter=subDomainCenter)
#                         pygame.draw.circle(surface, self.apoColor, screen_apo, 8)

            for terrainObject in environmentObj.terrainObjects:
                screenPos = self._toScreenCoords(terrainObject["pos"][0], terrainObject["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                radius = terrainObject["radius"] * self.scaleX
                if self.environment == "air":
                    pygame.draw.circle(surface, self.boulderColor, screenPos, max(5, radius))
                elif self.environment == "moon":
                    pygame.draw.circle(surface, self.moonBoulderColor, screenPos, max(5, radius))
                elif self.environment == "mars":
                    pygame.draw.circle(surface, self.marsBoulderColor, screenPos, max(5, radius))
            if self.environment == "air":
                for cloudObject in environmentObj.cloudObjects:
                    screenPos = self._toScreenCoords(cloudObject["pos"][0], cloudObject["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                    radius = cloudObject["radius"] * self.scaleX
                    pygame.draw.circle(surface, self.cloudColor, screenPos, max(5, radius))

        if self.environment == "air":
            for runwayObject in environmentObj.runwayObjects:
                screenPos = self._toScreenCoords(runwayObject["pos"][0], runwayObject["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.rect(surface, self.runwayColor, pygame.Rect(screenPos[0], screenPos[1], environmentObj.RUNWAY_SEGMENT_WIDTH * self.scaleX, environmentObj.RUNWAY_SEGMENT_HEIGHT * self.scaleX), 2)

        if self.environment == "space":
            for launchpadObject in environmentObj.launchpadObjects:
                screenPos = self._toScreenCoords(launchpadObject["pos"][0], launchpadObject["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.rect(surface, self.runwayColor, pygame.Rect(screenPos[0], screenPos[1], environmentObj.LAUNCHPAD_SEGMENT_WIDTH * self.scaleX, environmentObj.LAUNCHPAD_SEGMENT_HEIGHT * self.scaleX), 2)

        for obj in environmentObj.objects:
            orient = np.array(obj.orientationVector, dtype=float)
            norm = np.sqrt(orient @ orient)
            if norm == 0:
                continue
            localXAxis = orient / norm
            localZAxis = np.array([localXAxis[1], -localXAxis[0]])
            globalXCoords = []
            globalZCoords = []
            for xLocal, zLocal in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                globalPoint = xLocal * (-localXAxis) + zLocal * localZAxis + obj.positionVector
                globalXCoords.append(globalPoint[0])
                globalZCoords.append(globalPoint[1])
            centroidX = np.mean(globalXCoords)
            centroidZ = np.mean(globalZCoords)
            for x, z in zip(globalXCoords, globalZCoords):
                screenPos = self._toScreenCoords(x, z, isGlobalView=True, subDomainCenter=subDomainCenter)
                if (screenPos[0] > self.globalWidth + self.globalPos[0] or screenPos[0] < self.globalPos[0] or
                    screenPos[1] > self.globalHeight + self.globalPos[1] or screenPos[1] < self.globalPos[1]):
                    continue
                pygame.draw.circle(surface, self.pointColor, screenPos, 5)
            if self.SHOW_GLOBAL_FORCE_VECTOR:
                scale = 0.075
                force = np.array(obj.geometry.localForceVector, dtype=float)
                norm = np.sqrt(force[0]**2 + force[1]**2)
                if norm > 0:
                    force = force / norm * scale
                    forceGlobal = force[0] * localXAxis + force[1] * localZAxis
                    start = self._toScreenCoords(centroidX, centroidZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + forceGlobal[0], centroidZ + forceGlobal[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                    self._drawArrow(surface, start, end, self.forceColor)
            if self.environment in ["pilot", "air", "space", "moon", "mars"]:
                scale = 0.075
                thrust = obj.thrustForce
                norm = np.sqrt(thrust[0]**2 + thrust[1]**2)
                if norm > 0:
                    thrust = thrust / norm * scale
                    start = self._toScreenCoords(centroidX, centroidZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                    end = self._toScreenCoords(centroidX + thrust[0], centroidZ + thrust[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                    self._drawArrow(surface, start, end, self.thrustColor)
            if self.environment == "tunnel":
                globalVel = np.array(obj.velocityVector, dtype=float)
                globalVel = -globalVel
                norm = np.sqrt(globalVel[0]**2 + globalVel[1]**2)
                if norm > 0:
                    base_scale = 0.05
                    max_norm = 10.0
                    scale = base_scale * (min(norm, max_norm) / max_norm) * 5
                    globalVel = globalVel / norm * scale
                    xStart = -0.4
                    for offsetZ in [-0.2, 0.0, 0.2]:
                        start = self._toScreenCoords(xStart, offsetZ, isGlobalView=True, subDomainCenter=subDomainCenter)
                        end = self._toScreenCoords(xStart + globalVel[0], offsetZ + globalVel[1], isGlobalView=True, subDomainCenter=subDomainCenter)
                        self._drawArrow(surface, start, end, self.velocityColor, headSize=6)
        if self.environment == "pilot":
            for asteroid in environmentObj.asteroids:
                screenPos = self._toScreenCoords(asteroid["pos"][0], asteroid["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                radius = asteroid["radius"] * self.scaleX
                pygame.draw.circle(surface, self.asteroidColor, screenPos, max(5, radius))
            for laser in environmentObj.lasers:
                start = self._toScreenCoords(laser["pos"][0], laser["pos"][1], isGlobalView=True, subDomainCenter=subDomainCenter)
                end = self._toScreenCoords(laser["pos"][0] + laser["vel"][0] * 0.1, laser["pos"][1] + laser["vel"][1] * 0.1,
                                          isGlobalView=True, subDomainCenter=subDomainCenter)
                pygame.draw.line(surface, self.laserColor, start, end, 2)

    def _renderOrientationView(self, environmentObj, surface):
        overlay = pygame.Surface((self.orientWidth, self.orientHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        for obj in environmentObj.objects:
            xRange = max(obj.geometry.pointXCoords) - min(obj.geometry.pointXCoords)
            zRange = max(obj.geometry.pointZCoords) - min(obj.geometry.pointZCoords)
            maxRange = max(xRange, zRange, 1e-6)
            orientScale = min(self.orientWidth, self.orientHeight) / (2 * maxRange) * 0.8
            local_centroidX = np.mean(obj.geometry.pointXCoords)
            local_centroidZ = np.mean(obj.geometry.pointZCoords)
            if self.environment == "space":
                orient_vec = np.array(obj.orientationVector, dtype=float)
                norm = np.sqrt(orient_vec @ orient_vec)
                if norm == 0:
                    rotatedXCoords = list(obj.geometry.pointXCoords)
                    rotatedZCoords = list(obj.geometry.pointZCoords)
                    neg_lx0 = -1.0
                    neg_lx1 = 0.0
                    lz0 = 0.0
                    lz1 = 1.0
                else:
                    localXAxis = orient_vec / norm
                    localZAxis = np.array([localXAxis[1], -localXAxis[0]])
                    rotatedXCoords = []
                    rotatedZCoords = []
                    neg_lx0 = -localXAxis[0]
                    neg_lx1 = -localXAxis[1]
                    lz0 = localZAxis[0]
                    lz1 = localZAxis[1]
                    for xl, zl in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                        rx = xl * neg_lx0 + zl * lz0
                        rz = xl * neg_lx1 + zl * lz1
                        rotatedXCoords.append(rx)
                        rotatedZCoords.append(rz)
                centroidX = np.mean(rotatedXCoords)
                centroidZ = np.mean(rotatedZCoords)
                orientCenter = (centroidX, centroidZ)
                # Draw points
                for rx, rz in zip(rotatedXCoords, rotatedZCoords):
                    screenPos = self._toScreenCoords(rx, rz, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    pygame.draw.circle(overlay, self.pointColor, screenPos, 3)
                # Tangential velocities
                maxTangential = max(abs(min(obj.geometry.tangentialTotalVelocity)), abs(max(obj.geometry.tangentialTotalVelocity)), 1e-6)
                targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.25
                tangentialScale = (targetScreenLength / (maxTangential * orientScale)) if maxTangential > 0 else 0
                for i in range(len(obj.geometry.pointXCoords)):
                    xl, zl = obj.geometry.pointXCoords[i], obj.geometry.pointZCoords[i]
                    tangential = obj.geometry.tangentialTotalVelocity[i]
                    normalX, normalZ = obj.geometry.normalX[i], obj.geometry.normalZ[i]
                    lvec_x = -tangential * normalZ * tangentialScale
                    lvec_z = tangential * normalX * tangentialScale
                    rvec_x = lvec_x * neg_lx0 + lvec_z * lz0
                    rvec_z = lvec_x * neg_lx1 + lvec_z * lz1
                    rx = rotatedXCoords[i]
                    rz = rotatedZCoords[i]
                    end_rx = rx + rvec_x
                    end_rz = rz + rvec_z
                    start = self._toScreenCoords(rx, rz, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    end = self._toScreenCoords(end_rx, end_rz, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    self._drawArrow(surface, start, end, self.tangentialColor)
                # Force vector (obj.forceVector)
                force = np.array(obj.forceVector, dtype=float)
                norm_f = np.sqrt(force[0]**2 + force[1]**2)
                if norm_f > 0:
                    targetScreenLength_f = min(self.orientWidth, self.orientHeight) * 0.001
                    forceScale = targetScreenLength_f / norm_f
                    scaled_force = force * forceScale
                    start = self._toScreenCoords(centroidX, centroidZ, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    end = self._toScreenCoords(centroidX + scaled_force[0], centroidZ + scaled_force[1], isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    self._drawArrow(surface, start, end, self.forceColor, headSize=5)
                # Thrust vector (obj.thrustForce)
                thrust = np.array(obj.thrustForce, dtype=float)
                norm_t = np.sqrt(thrust[0]**2 + thrust[1]**2)
                if norm_t > 0:
                    targetScreenLength_t = min(self.orientWidth, self.orientHeight) * 0.001
                    thrustScale = targetScreenLength_t / norm_t
                    scaled_thrust = thrust * thrustScale
                    start = self._toScreenCoords(centroidX, centroidZ, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    end = self._toScreenCoords(centroidX + scaled_thrust[0], centroidZ + scaled_thrust[1], isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    self._drawArrow(surface, start, end, self.thrustColor, headSize=5)
                # Gravity vector (obj.gravityVector)
                gravity = np.array(obj.gravityVector, dtype=float)
                norm_g = np.sqrt(gravity[0]**2 + gravity[1]**2)
                if norm_g > 0:
                    targetScreenLength_g = min(self.orientWidth, self.orientHeight) * 0.001
                    gravityScale = targetScreenLength_g / norm_g
                    scaled_gravity = gravity * gravityScale
                    start = self._toScreenCoords(centroidX, centroidZ, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    end = self._toScreenCoords(centroidX + scaled_gravity[0], centroidZ + scaled_gravity[1], isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    self._drawArrow(surface, start, end, (0, 255, 0), headSize=5)  # Green color for gravity
            else:
                orientCenter = (local_centroidX, local_centroidZ)
                # Draw points
                for x, z in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                    screenPos = self._toScreenCoords(x, z, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    pygame.draw.circle(overlay, self.pointColor, screenPos, 3)
                # Draw lines
                for element in obj.geometry.connectionMatrix:
                    x1, z1 = obj.geometry.pointXCoords[element[0]], obj.geometry.pointZCoords[element[0]]
                    x2, z2 = obj.geometry.pointXCoords[element[1]], obj.geometry.pointZCoords[element[1]]
                    start = self._toScreenCoords(x1, z1, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    end = self._toScreenCoords(x2, z2, isOrientView=True, orientCenter=orientCenter, orientScale=orientScale)
                    pygame.draw.line(overlay, self.lineColor, start, end, 1)
                # Tangential velocities
                maxTangential = max(abs(min(obj.geometry.tangentialTotalVelocity)), abs(max(obj.geometry.tangentialTotalVelocity)), 1e-6)
                targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.25
                tangentialScale = (targetScreenLength / (maxTangential * orientScale)) if maxTangential > 0 else 0
                for i in range(len(obj.geometry.pointXCoords)):
                    x, z = obj.geometry.pointXCoords[i], obj.geometry.pointZCoords[i]
                    tangential = obj.geometry.tangentialTotalVelocity[i]
                    normalX, normalZ = obj.geometry.normalX[i], obj.geometry.normalZ[i]
                    vecX = -tangential * normalZ * tangentialScale
                    vecZ = tangential * normalX * tangentialScale
                    start = self._toScreenCoords(x, z, isOrientView=True, orientCenter=(local_centroidX, local_centroidZ), orientScale=orientScale)
                    end = self._toScreenCoords(x + vecX, z + vecZ, isOrientView=True, orientCenter=(local_centroidX, local_centroidZ), orientScale=orientScale)
                    self._drawArrow(surface, start, end, self.tangentialColor)
                # Local velocity vector
                localVel = np.array(obj.geometry.localVelocityVector, dtype=float)
                norm = np.sqrt(localVel[0]**2 + localVel[1]**2)
                if norm > 0:
                    unitVel = localVel / norm
                    targetScreenLength = 30
                    screenVel = unitVel * targetScreenLength
                    firstQuarterXStart = self.orientPos[0]
                    firstQuarterXEnd = self.orientPos[0] + self.orientWidth / 4
                    middleThirdYStart = self.orientPos[1] + self.orientHeight / 3
                    middleThirdYEnd = self.orientPos[1] + 2 * self.orientHeight / 3
                    xStartScreen = firstQuarterXStart + 10
                    yCenterScreen = (middleThirdYStart + middleThirdYEnd) / 2
                    xStart = (xStartScreen - (self.orientPos[0] + self.orientWidth / 2)) / orientScale + local_centroidX
                    screenSpacing = 10
                    offsetsScreen = [-screenSpacing, 0, screenSpacing]
                    for offsetY in offsetsScreen:
                        yStartScreen = yCenterScreen + offsetY
                        offsetZ = -((yStartScreen - (self.orientPos[1] + self.orientHeight / 2)) / orientScale) + local_centroidZ
                        start = self._toScreenCoords(xStart, offsetZ, isOrientView=True, orientCenter=(local_centroidX, local_centroidZ), orientScale=orientScale)
                        startScreen = list(start)
                        endScreen = [startScreen[0] + screenVel[0], startScreen[1] - screenVel[1]]
                        startScreen[0] = max(firstQuarterXStart, min(firstQuarterXEnd, startScreen[0]))
                        startScreen[1] = max(middleThirdYStart, min(middleThirdYEnd, startScreen[1]))
                        endScreen[0] = max(firstQuarterXStart, min(firstQuarterXEnd, endScreen[0]))
                        endScreen[1] = max(middleThirdYStart, min(middleThirdYEnd, endScreen[1]))
                        self._drawArrow(surface, startScreen, endScreen, self.velocityColor, headSize=5)
                # Force vector
                force = np.array(obj.geometry.localForceVector, dtype=float)
                norm = np.sqrt(force[0]**2 + force[1]**2)
                if norm > 0:
                    targetScreenLength = min(self.orientWidth, self.orientHeight) * 0.1
                    forceScale = targetScreenLength / (norm * orientScale)
                    force = force * forceScale
                    force[0] = -force[0]
                    startX = obj.geometry.pointXCoords[obj.geometry.numPoints // 4]
                    start = self._toScreenCoords(startX, 0, isOrientView=True, orientCenter=(local_centroidX, local_centroidZ), orientScale=orientScale)
                    end = self._toScreenCoords(startX + force[0], force[1], isOrientView=True, orientCenter=(local_centroidX, local_centroidZ), orientScale=orientScale)
                    self._drawArrow(surface, start, end, self.forceColor, headSize=5)
        surface.blit(overlay, self.orientPos)

    def _renderMinimapView(self, environmentObj, surface):
        overlay = pygame.Surface((self.minimapWidth, self.minimapHeight), pygame.SRCALPHA)
        overlay.fill(self.orientBgColor)
        if self.environment in ["ocean", "pilot"]:
            pygame.draw.line(overlay, self.oceanColor, (0, 0), (self.minimapWidth, 0), 2)
            pygame.draw.line(overlay, self.oceanColor, (0, self.minimapHeight), (self.minimapWidth, self.minimapHeight), 2)
            pygame.draw.line(overlay, self.oceanColor, (0, 0), (0, self.minimapHeight), 2)
            pygame.draw.line(overlay, self.oceanColor, (self.minimapWidth, 0), (self.minimapWidth, self.minimapHeight), 2)
        elif self.environment in ["air", "moon", "mars"]:
            pygame.draw.line(overlay, self.oceanColor, (0, 0), (self.minimapWidth, 0), 2)
            pygame.draw.line(overlay, self.oceanColor, (0, self.minimapHeight), (self.minimapWidth, self.minimapHeight), 2)
            pygame.draw.line(overlay, self.oceanColor, (0, 0), (0, self.minimapHeight), 2)
            pygame.draw.line(overlay, self.oceanColor, (self.minimapWidth, 0), (self.minimapWidth, self.minimapHeight), 2)
        elif self.environment == "space":
            # Draw Earth in minimap
            earth_center = self._toScreenCoords(0, 0, isMinimapView=True)
            scale = min(self.minimapWidth, self.minimapHeight) / (3 * max(self.environmentObj.earthRadius, getattr(environmentObj.objects[0], 'semimajor', self.environmentObj.earthRadius) * (1 + getattr(environmentObj.objects[0], 'eccentricity', 0))))
            earth_radius_screen = self.environmentObj.earthRadius * scale
            pygame.draw.circle(overlay, self.groundColor, earth_center, earth_radius_screen)
            # Orbital trajectory in minimap
            if environmentObj.objects:
                obj = environmentObj.objects[0]
                a = getattr(obj, 'semimajor', 0)
                e = getattr(obj, 'eccentricity', 0)
                nu = getattr(obj, 'trueanomaly', 0)
                pos = np.array(obj.positionVector)
                r = np.linalg.norm(pos)
                if r > 0 and a > 0:
                    phi = math.atan2(pos[1], pos[0])
                    omega = phi - nu
                    num_points = 100
                    trajectory = []
                    for i in range(num_points + 1):
                        theta = 2 * math.pi * i / num_points
                        r_theta = a * (1 - e**2) / (1 + e * math.cos(theta))
                        tx = r_theta * math.cos(theta + omega)
                        tz = r_theta * math.sin(theta + omega)
                        trajectory.append((tx, tz))
                    prev_screen = None
                    for px, pz in trajectory:
                        screen_pos = self._toScreenCoords(px, pz, isMinimapView=True)
                        if 0 <= screen_pos[0] <= self.minimapWidth and 0 <= screen_pos[1] <= self.minimapHeight:
                            if prev_screen is not None:
                                pygame.draw.line(overlay, self.orbitColor, prev_screen, screen_pos, 1)
                            prev_screen = screen_pos
                    # Periapsis point in minimap
                    r_peri = a * (1 - e)
                    theta_peri = 0
                    px_peri = r_peri * math.cos(theta_peri + omega)
                    pz_peri = r_peri * math.sin(theta_peri + omega)
                    screen_peri = self._toScreenCoords(px_peri, pz_peri, isMinimapView=True)
                    if 0 <= screen_peri[0] <= self.minimapWidth and 0 <= screen_peri[1] <= self.minimapHeight:
                        pygame.draw.circle(overlay, self.periColor, screen_peri, 3)
                    # Apoapsis point in minimap
                    r_apo = a * (1 + e)
                    theta_apo = math.pi
                    px_apo = r_apo * math.cos(theta_apo + omega)
                    pz_apo = r_apo * math.sin(theta_apo + omega)
                    screen_apo = self._toScreenCoords(px_apo, pz_apo, isMinimapView=True)
                    if 0 <= screen_apo[0] <= self.minimapWidth and 0 <= screen_apo[1] <= self.minimapHeight:
                        pygame.draw.circle(overlay, self.apoColor, screen_apo, 3)

        for obj in environmentObj.objects:
            # Transform local geometry points to global coordinates
            orient = np.array(obj.orientationVector, dtype=float)
            norm = np.sqrt(orient @ orient)
            if norm == 0:
                localXAxis = np.array([1.0, 0.0])
                localZAxis = np.array([0.0, 1.0])
            else:
                localXAxis = orient / norm
                localZAxis = np.array([localXAxis[1], -localXAxis[0]])
            globalXCoords = []
            globalZCoords = []
            for xLocal, zLocal in zip(obj.geometry.pointXCoords, obj.geometry.pointZCoords):
                globalPoint = xLocal * (-localXAxis) + zLocal * localZAxis + obj.positionVector
                globalXCoords.append(globalPoint[0])
                globalZCoords.append(globalPoint[1])
            
            # Draw geometry points
            for x, z in zip(globalXCoords, globalZCoords):
                if x == 0 and z == 0:
                    continue
                screenPos = self._toScreenCoords(x, z, isMinimapView=True)
                if (0 <= screenPos[0] <= self.minimapWidth and 
                    0 <= screenPos[1] <= self.minimapHeight):
                    pygame.draw.circle(overlay, self.pointColor, screenPos, 1)
            
            # Draw connection lines
            for element in obj.geometry.connectionMatrix:
                x1, z1 = globalXCoords[element[0]], globalZCoords[element[0]]
                x2, z2 = globalXCoords[element[1]], globalZCoords[element[1]]
                if (x1 == 0 and z1 == 0) or (x2 == 0 and z2 == 0):
                    continue
                start = self._toScreenCoords(x1, z1, isMinimapView=True)
                end = self._toScreenCoords(x2, z2, isMinimapView=True)
                pygame.draw.line(overlay, self.lineColor, start, end, 1)
            
            # Draw path history
            if hasattr(environmentObj, 'pathHistory'):
                for pos in environmentObj.pathHistory:
                    if pos[0] == 0 and pos[1] == 0:
                        continue
                    screenPos = self._toScreenCoords(pos[0], pos[1], isMinimapView=True)
                    if (0 <= screenPos[0] <= self.minimapWidth and 
                        0 <= screenPos[1] <= self.minimapHeight):
                        pygame.draw.circle(overlay, (255, 0, 0), screenPos, 1)
            
            # Draw spacecraft marker using positionVector
            markerPos = self._toScreenCoords(obj.positionVector[0], obj.positionVector[1], isMinimapView=True)
            if (0 <= markerPos[0] <= self.minimapWidth and 
                0 <= markerPos[1] <= self.minimapHeight and 
                not (obj.positionVector[0] == 0 and obj.positionVector[1] == 0)):
                pygame.draw.circle(overlay, (255, 105, 180), markerPos, 3)
        
        if self.environment == "pilot":
            for asteroid in environmentObj.asteroids:
                screenPos = self._toScreenCoords(asteroid["pos"][0], asteroid["pos"][1], isMinimapView=True)
                if (0 <= screenPos[0] <= self.minimapWidth and 
                    0 <= screenPos[1] <= self.minimapHeight):
                    radius = asteroid["radius"] * (self.minimapWidth / self.deltaX)
                    pygame.draw.circle(overlay, self.asteroidColor, screenPos, max(2, radius))
        
        surface.blit(overlay, self.minimapPos)

    def quit(self):
        pygame.quit()