import pygame
import numpy as np
import math
import random

class UIManager:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.digitHeight = 20
        self.digitFont = pygame.font.SysFont("courier", 16, bold=True)
        self.labelFont = pygame.font.SysFont("ocraextended", 40, bold=False)
        self.textColor = (0, 0, 0)
        self.digitBgColor = (50, 50, 50)
        self.digitColor = (255, 255, 255)
        self.digitWindowColor = (30, 30, 30)
        self.orientBgColor = (200, 200, 200, 128)
        self.hiddenUIShown = False
        self.debugReadoutsShown = False
        self.debugHistory = {}
        self.debugSurface = None
        self.stars = []
        self.tunnelParams = {
            "velocity": 10.0,  # Maximum velocity
            "alpha": 0.0,     # Angle of attack
            "unsteady": False,
            "T": 100.0        # Period of oscillation
        }
        self.oceanParams = {
            "deltaX": 100.0,
            "deltaZ": 100.0
        }
        self.pilotParams = {
            "deltaX": 100.0,
            "deltaZ": 100.0
        }
        self.airParams = {
            "deltaX": 100000.0,
            "deltaZ": 2000.0,
            "enginePower": 2000,
            "deltaRotation": 0.1,
            "rotationMin": -15,
            "rotationMax": 30,
        }
        self.spaceParams = {
            "deltaX": 1000.0,
            "deltaZ": 10000.0,
            "enginePower": 2000,
            "deltaRotation": 0.1,
            "rotationMin": -45,
            "rotationMax": 45,
        }
        self.uiElements = []
        self.environment = None
        self._updateDimensions(windowWidth, windowHeight)
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()
        self.wheelOffsets = {}
        self.animationSpeed = 15
        self.animationDuration = 10
        self.animationFrames = {}
        self.prevValues = {}
        self.prevAoa = {}
        self.smoothingFactor = 0.1
        self.frameNumber = 0
        self.screen = pygame.display.get_surface()

    def _createUIElements(self):
        elements = []
        if self.hiddenUIShown:
            y = self.uiPanelRect.y + 10
            spacing = 40
            if self.environment == "tunnel":
                # Unsteady checkbox
                unsteadyRect = pygame.Rect(self.uiPanelRect.x + 10, y, 25, 25)
                elements.append({"type": "checkbox", "label": "Unst:", "value": self.tunnelParams["unsteady"], "rect": unsteadyRect, "key": "unsteady"})
                y += spacing
                # Base fields: velocity and alpha
                fields = [
                    ("Vel:", str(self.tunnelParams["velocity"]), "velocity"),
                    ("α:", str(self.tunnelParams["alpha"]), "alpha"),
                ]
                # Add period (T) field only if unsteady is True
                if self.tunnelParams["unsteady"]:
                    unsteadyFields = [
                        ("T:", str(self.tunnelParams["T"]), "T"),
                    ]
                    fields.extend(unsteadyFields)
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
            elif self.environment == "ocean" or self.environment == "pilot":
                fields = [
                    ("ΔX:", str(self.oceanParams["deltaX"]), "deltaX"),
                    ("ΔZ:", str(self.oceanParams["deltaZ"]), "deltaZ")
                ]
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
            elif self.environment == "air":
                fields = [
                    ("ΔX:", str(self.airParams["deltaX"]), "deltaX"),
                    ("ΔZ:", str(self.airParams["deltaZ"]), "deltaZ"),
                    ("Pwr:", str(self.airParams["enginePower"]), "enginePower"),
                    ("Δθ:", str(self.airParams["deltaRotation"]), "deltaRotation"),
                    ("- θ:", str(self.airParams["rotationMin"]), "rotationMin"),
                    ("+θ:", str(self.airParams["rotationMax"]), "rotationMax")
                ]
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
            elif self.environment == "space":
                fields = [
                    ("ΔX:", str(self.spaceParams["deltaX"]), "deltaX"),
                    ("ΔZ:", str(self.spaceParams["deltaZ"]), "deltaZ"),
                    ("Pwr:", str(self.spaceParams["enginePower"]), "enginePower"),
                    ("Δθ:", str(self.spaceParams["deltaRotation"]), "deltaRotation"),
                    ("- θ:", str(self.spaceParams["rotationMin"]), "rotationMin"),
                    ("+θ:", str(self.spaceParams["rotationMax"]), "rotationMax")
                ]
                for label, value, key in fields:
                    rect = pygame.Rect(self.uiPanelRect.x + 10, y, 150, 25)
                    elements.append({"type": "textbox", "label": label, "value": value, "rect": rect, "key": key, "active": False})
                    y += spacing
        return elements

    def _updateDimensions(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.readoutWidth = windowWidth // 6
        self.readoutHeight = windowHeight
        debug_width = 410 if self.debugReadoutsShown else 0
        self.readoutPos = (windowWidth - self.readoutWidth, 0)  # Keep readout on right
        self.digitWidth = (self.readoutWidth - 60) // 4
        self.uiPanelRect = pygame.Rect(450, 50, 300, 400)  # Move UI panel to avoid overlap with debug
        self.uiElements = self._createUIElements()
        self.stars = []

    def resize(self, size):
        self._updateDimensions(*size)
        self.digitWheels = [self._createDigitWheel() for _ in range(3)]
        self.signWheel = self._createSignWheel()

    def _createDigitWheel(self):
        wheelHeight = self.digitHeight * 10
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i in range(10):
            digitText = self.digitFont.render(str(i), True, self.digitColor)
            xPos = (self.digitWidth - digitText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - digitText.get_height()) // 2
            wheel.blit(digitText, (xPos, yPos))
        return wheel

    def _createSignWheel(self):
        wheelHeight = self.digitHeight * 2
        wheel = pygame.Surface((self.digitWidth, wheelHeight), pygame.SRCALPHA)
        for i, symbol in enumerate(["+", "-"]):
            symbolText = self.digitFont.render(symbol, True, self.digitColor)
            xPos = (self.digitWidth - symbolText.get_width()) // 2
            yPos = i * self.digitHeight + (self.digitHeight - symbolText.get_height()) // 2
            wheel.blit(symbolText, (xPos, yPos))
        return wheel

    def toggleHiddenUI(self):
        self.hiddenUIShown = not self.hiddenUIShown
        self.uiElements = self._createUIElements()
        
    def toggleDebugReadouts(self):
        self.debugReadoutsShown = not self.debugReadoutsShown
        if self.debugReadoutsShown:
            self.debugSurface = pygame.Surface((400, 720), pygame.SRCALPHA)  # Initialize with correct size
        else:
            self.debugSurface = None
        self._updateDimensions(self.windowWidth, self.windowHeight)
        
    def handleClick(self, pos):
        x, y = pos
        for element in self.uiElements:
            if element["rect"].collidepoint(x, y):
                if element["type"] == "textbox":
                    element["active"] = True
                elif element["type"] == "checkbox":
                    if element["key"] == "unsteady":
                        self.tunnelParams["unsteady"] = not self.tunnelParams["unsteady"]
                        element["value"] = self.tunnelParams["unsteady"]
                        self.uiElements = self._createUIElements()
                for other in self.uiElements:
                    if other != element and other["type"] == "textbox":
                        other["active"] = False
        return False
    
    def handleKey(self, event):
        for element in self.uiElements:
            if element.get("active", False) and element["type"] == "textbox":
                if event.key == pygame.K_BACKSPACE:
                    element["value"] = element["value"][:-1]
                elif event.key == pygame.K_RETURN:
                    element["active"] = False
                    try:
                        value = float(element["value"])
                        if self.environment == "tunnel":
                            self.tunnelParams[element["key"]] = value
                        if self.environment == "ocean":
                            self.oceanParams[element["key"]] = value
                        if self.environment == "pilot":
                            self.pilotParams[element["key"]] = value
                        if self.environment == "air":
                            self.airParams[element["key"]] = value
                        if self.environment == "space":
                            self.spaceParams[element["key"]] = value
                    except ValueError:
                        if self.environment == "tunnel":
                            element["value"] = self.tunnelParams.get(element["key"], 0.0)
                        if self.environment == "ocean":
                            element["value"] = self.oceanParams.get(element["key"], 0.0)
                        if self.environment == "pilot":
                            element["value"] = self.pilotParams.get(element["key"], 0.0)
                        if self.environment == "air":
                            element["value"] = self.airParams.get(element["key"], 0.0)
                        if self.environment == "space":
                            element["value"] = self.spaceParams.get(element["key"], 0.0)
                elif event.unicode.isprintable():
                    element["value"] += event.unicode

    def getTunnelFlowParams(self):
        velocity = float(self.tunnelParams["velocity"])
        alphaDeg = float(self.tunnelParams["alpha"])
        unsteady = self.tunnelParams["unsteady"]
        T = float(self.tunnelParams["T"])
#         velocity = 10.0
#         alphaDeg = 0.0
#         unsteady = True
#         T = 100
        return velocity, alphaDeg, unsteady, T

    def getOceanParams(self):
        return self.oceanParams["deltaX"], self.oceanParams["deltaZ"]
        
    def getPilotParams(self):
        return self.pilotParams["deltaX"], self.pilotParams["deltaZ"]
    
    def getAirParams(self):
        return self.airParams["deltaX"], self.airParams["deltaZ"], self.airParams["enginePower"], self.airParams["deltaRotation"], self.airParams["rotationMin"], self.airParams["rotationMax"]
        
    def getSpaceParams(self):
        return self.spaceParams["deltaX"], self.spaceParams["deltaZ"], self.spaceParams["enginePower"], self.spaceParams["deltaRotation"], self.spaceParams["rotationMin"], self.spaceParams["rotationMax"]
    
    def renderShopUI(self, shop):
        pass

    def renderEnvironmentUI(self, environmentObj, envType):
        self.environment = envType
        self.frameNumber = environmentObj.frameNumber
        self._renderReadoutView(environmentObj, self.screen)
        if self.debugReadoutsShown:
            self.debugSurface = pygame.Surface((400, 720), pygame.SRCALPHA)  # Match _renderDebugSurface size
            self._renderDebugSurface(environmentObj, self.debugSurface)
            blit_pos = (10, max(0, (self.windowHeight - 720) // 2 - 20))  # Center left, adjusted up
#             print(f"Blit pos: {blit_pos}, Window size: ({self.windowWidth}, {self.windowHeight})")
            self.screen.blit(self.debugSurface, blit_pos)
        if self.hiddenUIShown:
            pygame.draw.rect(self.screen, self.orientBgColor, self.uiPanelRect, border_radius=5)
            smallFont = pygame.font.SysFont("ocraextended", 20, bold=False)
            for element in self.uiElements:
                if element["type"] == "textbox":
                    if element.get("active", False):
                        pygame.draw.rect(self.screen, (255, 165, 0), element["rect"], 2, border_radius=3)
                    pygame.draw.rect(self.screen, (220, 220, 220), element["rect"], border_radius=3)
                    label = smallFont.render(element["label"], True, self.textColor)
                    self.screen.blit(label, (element["rect"].x - 50, element["rect"].y + 5))
                    value = smallFont.render(element["value"], True, self.textColor)
                    self.screen.blit(value, (element["rect"].x + 5, element["rect"].y + 5))
                elif element["type"] == "checkbox":
                    pygame.draw.rect(self.screen, (220, 220, 220), element["rect"], border_radius=3)
                    if element["value"]:
                        pygame.draw.line(self.screen, (0, 0, 0), 
                                         (element["rect"].x + 5, element["rect"].y + 12), 
                                         (element["rect"].x + 12, element["rect"].y + 20), 2)
                        pygame.draw.line(self.screen, (0, 0, 0), 
                                         (element["rect"].x + 12, element["rect"].y + 20), 
                                         (element["rect"].x + 20, element["rect"].y + 5), 2)
                    label = smallFont.render(element["label"], True, self.textColor)
                    self.screen.blit(label, (element["rect"].x - 50, element["rect"].y + 5))

    def _renderDebugSurface(self, environmentObj, surface):
        # Use a surface height of 720 to fit all content
        if surface is None or surface.get_size() != (400, 720):
            surface = pygame.Surface((400, 720), pygame.SRCALPHA)
        surface.fill((50, 50, 50))  # Dark gray background
        pygame.draw.rect(surface, (255, 255, 255), (0, 0, 400, 720), 2)  # White border
        self.debugSurface = surface
    
        if len(environmentObj.objects) == 0:
            text = pygame.font.SysFont("courier", 14).render("No objects", True, (255, 255, 255))
            surface.blit(text, (10, 10))
            return
    
        obj = environmentObj.objects[0]
        obj_id = id(obj)
        if obj_id not in self.debugHistory:
            self.debugHistory[obj_id] = {
                'position': [], 'position_z': [], 'velocity': [], 'acceleration': [], 'airspeed_body': [],
                'orientation': [], 'rotAngle': [], 'aoa': [], 'throttle': [], 'thrust': [],
                'force': [], 'lift': [], 'drag': [], 'addedMass': []
            }
        history = self.debugHistory[obj_id]
        max_history = 100
    
        # Update history
        pos = tuple(obj.positionVector)
        history['position'].append(pos)
        history['position_z'].append(pos[1])  # Store z-component separately
        vel = tuple(obj.velocityVector)
        history['velocity'].append(vel)
        acc = tuple(obj.accelerationVector)
        history['acceleration'].append(acc)
    
        orient = np.array(obj.orientationVector, dtype=float)
        history['orientation'].append(tuple(orient))
    
        # Compute rotAngle
        refVector = np.array([0, -1], dtype=float)
        normOrient = np.linalg.norm(orient)
        normRef = np.linalg.norm(refVector)
        rotAngle = 0.0
        if normOrient > 0 and normRef > 0:
            cosTheta = np.dot(orient, refVector) / (normOrient * normRef)
            cosTheta = np.clip(cosTheta, -1.0, 1.0)
            rotAngle = math.degrees(math.acos(cosTheta))
            cross = orient[0] * refVector[1] - orient[1] * refVector[0]
            if cross < 0:
                rotAngle = -rotAngle
            rotAngle = ((rotAngle + 180) % 360) - 180
        history['rotAngle'].append(rotAngle)
    
        # Compute aoa
        vel_arr = np.array(obj.velocityVector, dtype=float)
        normVel = np.linalg.norm(vel_arr)
        aoa = 0.0
        if normOrient > 0 and normVel > 0:
            orientNorm = orient / normOrient
            velNorm = vel_arr / normVel
            cosTheta = np.dot(orientNorm, velNorm)
            cosTheta = np.clip(cosTheta, -1.0, 1.0)
            aoa = math.degrees(math.acos(cosTheta))
            cross = orientNorm[0] * velNorm[1] - orientNorm[1] * velNorm[0]
            if cross < 0:
                aoa = -aoa
            aoa = ((aoa + 180) % 360) - 180
            key = f"{obj_id}_AoA"
            if key in self.prevAoa:
                aoa = self.prevAoa[key] + self.smoothingFactor * (aoa - self.prevAoa[key])
            self.prevAoa[key] = aoa
        history['aoa'].append(aoa)
    
        # Airspeed body
        body_airspeed = 0.0
        if normOrient > 0:
            body_airspeed = np.dot(vel_arr, orient / normOrient)
        history['airspeed_body'].append(body_airspeed)
    
        # Throttle
        throttle = obj.engineThrottle * 100 if hasattr(obj, 'engineThrottle') and obj.engineThrottle <= 1.0 else (getattr(obj, 'engineThrottle', 0) or 0)
        history['throttle'].append(throttle)
    
        # Thrust
        thrust = tuple(getattr(obj, 'thrustForce', [0, 0]))
        history['thrust'].append(thrust)
    
        # Force
        force = tuple(getattr(obj, 'forceVector', [0, 0]))
        history['force'].append(force)
    
        # Lift and Drag
        thrust_arr = np.array(thrust)
        force_arr = np.array(force)
        aero = force_arr - thrust_arr
        drag = 0.0
        lift_signed = 0.0
        if normOrient > 0:
            orient_norm = orient / normOrient
            drag = np.dot(aero, orient_norm)
            perp = np.array([-orient_norm[1], orient_norm[0]])
            lift_signed = np.dot(aero, perp)
        history['drag'].append(drag)
        history['lift'].append(lift_signed)
    
        # Added mass
        added = getattr(obj, 'addedMass', 0.0)
        history['addedMass'].append(added)
    
        # Trim history
        for key in history:
            if len(history[key]) > max_history:
                history[key].pop(0)
    
        # Render to surface
        font = pygame.font.SysFont("courier", 12)
        small_font = pygame.font.SysFont("courier", 10)
    
        def render_value(label, value, y_pos):
            text = font.render(f"{label}: {value}", True, (255, 255, 255))
            surface.blit(text, (10, y_pos))
            return y_pos + 12  # Reduced spacing
    
        def render_graph(hist_key, y_pos, is_vector=False):
            text = small_font.render(f"Graph: {hist_key}", True, (200, 200, 200))
            surface.blit(text, (10, y_pos))
            y_pos += 10  # Reduced spacing
            graph_rect = pygame.Rect(10, y_pos, 380, 20)  # Reduced graph height
            pygame.draw.rect(surface, (30, 30, 30), graph_rect)
            pygame.draw.rect(surface, (100, 100, 100), graph_rect, 1)
            h = history[hist_key]
            if len(h) < 2:
                return y_pos + 25
            if is_vector:
                vals = [np.linalg.norm(np.array(v)) for v in h]
            else:
                vals = [float(v) for v in h if isinstance(v, (int, float))]
            if not vals:
                return y_pos + 25
            minv = min(vals)
            maxv = max(vals)
            rangev = maxv - minv if maxv != minv else 1.0
            graph_w = 380
            graph_h = 20
            prev_x = None
            prev_y = None
            n = len(vals)
            for i, v in enumerate(vals):
                x = 10 + (i / max(n - 1, 1)) * graph_w
                yg = y_pos + graph_h - ((v - minv) / rangev * graph_h)
                if prev_x is not None:
                    pygame.draw.line(surface, (0, 255, 0), (prev_x, prev_y), (x, yg), 1)
                prev_x = x
                prev_y = yg
            return y_pos + 25  # Reduced spacing
    
        def render_position_z_graph(y_pos):
            text = small_font.render("Graph: Position Z", True, (200, 200, 200))
            surface.blit(text, (10, y_pos))
            y_pos += 10
            graph_rect = pygame.Rect(10, y_pos, 380, 20)
            pygame.draw.rect(surface, (30, 30, 30), graph_rect)
            pygame.draw.rect(surface, (100, 100, 100), graph_rect, 1)
            h = history['position_z']
            if len(h) < 2:
                return y_pos + 25
            vals = [float(v) for v in h]
            if not vals:
                return y_pos + 25
            minv = min(vals)
            maxv = max(vals)
            rangev = maxv - minv if maxv != minv else 1.0
            graph_w = 380
            graph_h = 20
            prev_x = None
            prev_y = None
            n = len(vals)
            for i, v in enumerate(vals):
                x = 10 + (i / max(n - 1, 1)) * graph_w
                yg = y_pos + graph_h - ((v - minv) / rangev * graph_h)
                if prev_x is not None:
                    pygame.draw.line(surface, (0, 255, 0), (prev_x, prev_y), (x, yg), 1)
                prev_x = x
                prev_y = yg
            return y_pos + 25
    
        y = 10
        # Numerical values for position and orientation
        y = render_value("Orientation", f"({orient[0]:.2f}, {orient[1]:.2f})", y)
        y = render_value("Position", f"({pos[0]:.2f}, {pos[1]:.2f})", y)
        y = render_graph('position_z', y, False)
        # Other values with graphs
        y = render_value("Velocity", f"({vel[0]:.2f}, {vel[1]:.2f})", y)
        y = render_graph('velocity', y, True)
        y = render_value("Acceleration", f"({acc[0]:.2f}, {acc[1]:.2f})", y)
        y = render_graph('acceleration', y, True)
        y = render_value("Airspeed (body)", f"{body_airspeed:.2f}", y)
        y = render_graph('airspeed_body', y)
        y = render_value("Rot Angle", f"{rotAngle:.2f} deg", y)
        y = render_graph('rotAngle', y)
        y = render_value("AoA", f"{aoa:.2f} deg", y)
        y = render_graph('aoa', y)
        y = render_value("Throttle", f"{throttle:.1f}%", y)
        y = render_graph('throttle', y)
        y = render_value("Thrust Force", f"({thrust[0]:.2f}, {thrust[1]:.2f})", y)
        y = render_graph('thrust', y, True)
        y = render_value("Force Vector", f"({force[0]:.2f}, {force[1]:.2f})", y)
        y = render_graph('force', y, True)
        y = render_value("Lift (signed)", f"{lift_signed:.2f}", y)
        y = render_graph('lift', y)
        y = render_value("Drag", f"{drag:.2f}", y)
        y = render_graph('drag', y)
        y = render_value("Added Mass", f"{added:.2f}", y)
        y = render_graph('addedMass', y)

    def _renderReadoutView(self, environmentObj, surface):
        overlay = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))
        yPos = self.readoutPos[1] + 10
        if self.environment in ["ocean"]:
            labels = ["POS X:", "POS Z:", "VEL X:", "VEL Z:", "ROT ANG:", "AOA:"]
        elif self.environment in ["pilot"]:
            labels = ["POS X:", "POS Z:", "VEL X:", "VEL Z:", "THRUST:", "ROT ANG:", "AOA:"]
        elif self.environment in ["air", "space"]:
            labels = ["SPEED:", "ALT:", "AIRSPD:", "PWR", "AOA:"]
        else:  # tunnel
            labels = ["VEL X:", "VEL Z:", "ACC X:", "ACC Z:", "ROT ANG:", "AOA:"]
        spacing = (self.readoutHeight - 20) // len(labels)
        wheel_offset = 5
        if self.environment in ["air", "space"]:
            # Altitude indicator
            if len(environmentObj.objects) > 0:
                obj = environmentObj.objects[0]
                deltaZ = self.airParams["deltaZ"] if self.environment == "air" else self.spaceParams["deltaZ"]
                altitude = obj.positionVector[1]
                bar_x = 5
                bar_y = 50
                bar_width = 25
                bar_height = self.readoutHeight - 100
                num_layers = 6
                layer_h = bar_height / num_layers
                # Colors: top (space) black to bottom (ground) light blue
                colors = [(0, 0, 0), (0, 0, 139), (25, 25, 112), (70, 130, 180), (100, 149, 237), (135, 206, 235)]
                for i in range(num_layers):
                    col = colors[i]
                    yy = bar_y + i * layer_h
                    pygame.draw.rect(overlay, col, (bar_x, yy, bar_width, layer_h))
                # Stars in top layer (space, i=0)
                top_y_start = bar_y
                if len(self.stars) == 0:
                    for _ in range(15):
                        self.stars.append((random.randint(0, bar_width), random.randint(0, int(layer_h))))
                for sx, sy in self.stars:
                    pygame.draw.circle(overlay, (255, 255, 255), (int(bar_x + sx), int(top_y_start + sy)), 1)
                # Arrow - base on left edge, pointing right into bar
                if deltaZ > 0:
                    arrow_frac = min(max(altitude / deltaZ, 0), 1)
                    arrow_y = bar_y + bar_height * (1 - arrow_frac)
                    points = [(bar_x, arrow_y - 5), (bar_x + 10, arrow_y), (bar_x, arrow_y + 5)]
                    pygame.draw.polygon(overlay, (255, 255, 255), points)
            wheel_offset = bar_x + bar_width + 20  # Adjust spacing
        for obj in environmentObj.objects:
            if self.environment == "ocean":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                values = [posX, posZ, velX, velZ]
            elif self.environment == "pilot":
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                thrust = np.linalg.norm(obj.thrustForce)
                values = [posX, posZ, velX, velZ, thrust]
            elif self.environment in ["air", "space"]:
                posX, posZ = obj.positionVector[0], obj.positionVector[1]
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                ground_speed = abs(velX)
                altitude = posZ
                airspeed = np.sqrt(velX**2 + velZ**2)
                throttle = obj.engineThrottle * 100 if obj.engineThrottle <= 1.0 else obj.engineThrottle
                values = [ground_speed, altitude, airspeed, throttle]
            elif self.environment == "tunnel":
                velX, velZ = obj.velocityVector[0], obj.velocityVector[1]
                accX, accZ = obj.accelerationVector[0], obj.accelerationVector[1]
                velX, velZ = -velX, -velZ
                accX, accZ = -accX, -accZ
                accX = accX * 10
                accZ = accZ * 10
                values = [velX, velZ, accX, accZ]
            orient = np.array(obj.orientationVector, dtype=float)
            refVector = np.array([0, -1], dtype=float)
            normOrient = np.linalg.norm(orient)
            normRef = np.linalg.norm(refVector)
            if normOrient > 0 and normRef > 0:
                cosTheta = np.dot(orient, refVector) / (normOrient * normRef)
                cosTheta = np.clip(cosTheta, -1.0, 1.0)
                rotAngle = math.degrees(math.acos(cosTheta))
                cross = orient[0] * refVector[1] - orient[1] * refVector[0]
                if cross < 0:
                    rotAngle = -rotAngle
                rotAngle = ((rotAngle + 180) % 360) - 180
            else:
                rotAngle = 0.0
            if self.environment not in ["air", "space"]:
                values.append(rotAngle)
            vel = np.array(obj.velocityVector, dtype=float)
            normVel = np.linalg.norm(vel)
            normOrient = np.linalg.norm(orient)
            if normOrient > 0 and normVel > 0:
                orientNorm = orient / normOrient
                velNorm = vel / normVel
                cosTheta = np.dot(orientNorm, velNorm)
                cosTheta = np.clip(cosTheta, -1.0, 1.0)
                aoa = math.degrees(math.acos(cosTheta))
                cross = orientNorm[0] * velNorm[1] - orientNorm[1] * velNorm[0]
                if cross < 0:
                    aoa = -aoa
                aoa = ((aoa + 180) % 360) - 180
                key = f"{id(obj)}_AoA"
                if key in self.prevAoa:
                    aoa = self.prevAoa[key] + self.smoothingFactor * (aoa - self.prevAoa[key])
                self.prevAoa[key] = aoa
            else:
                aoa = 0.0
            values.append(aoa)
            for idx, (label, value) in enumerate(zip(labels, values)):
                key = f"{id(obj)}_{label}_value"
                if key not in self.prevValues:
                    self.prevValues[key] = value
                smoothedValue = self.prevValues[key] + self.smoothingFactor * (value - self.prevValues[key])
                self.prevValues[key] = smoothedValue
                value = smoothedValue
                numDigits = 3
                if label in ["AOA:"]:
                    maxValue = 180
                    minValue = -180
                    value = max(min(value, maxValue), minValue)
                    valueStr = f"{abs(int(value)):03d}"
                else:
                    maxValue = 999
                    minValue = -999
                    value = max(min(value, maxValue), minValue)
                    absValue = abs(value)
                    hundreds = int(absValue // 100) % 10
                    tens = int(absValue // 10) % 10
                    ones = int(absValue) % 10
                    valueStr = f"{hundreds:01d}{tens:01d}{ones:01d}"
                signIdx = 0 if value >= 0 else 1
                totalWheels = 4
                key = f"{id(obj)}_{label}"
                if key not in self.wheelOffsets:
                    self.wheelOffsets[key] = [0] * totalWheels
                if key not in self.animationFrames:
                    self.animationFrames[key] = [0] * totalWheels
                labelText = self.labelFont.render(label, True, self.textColor)
                labelX = wheel_offset
                labelY = yPos + 2 - self.readoutPos[1]
                overlay.blit(labelText, (labelX, labelY))
                windowY = yPos + 50 - self.readoutPos[1]
                windowX = wheel_offset
                windowWidth = self.digitWidth * totalWheels
                pygame.draw.rect(overlay, self.digitWindowColor, (windowX, windowY, windowWidth, self.digitHeight))
                if self.environment == "tunnel" and label in ["ACC X:", "ACC Z:"]:
                    decimalX = windowX + self.digitWidth * 3
                    decimalY = windowY + self.digitHeight - 3
                    pygame.draw.circle(overlay, self.digitColor, (decimalX, decimalY), 1)
                wheelsToDraw = [(self.signWheel, signIdx, 2)]
                for i, digit in enumerate(valueStr):
                    wheelsToDraw.append((self.digitWheels[i], int(digit), 10))
                for i, (wheel, targetDigit, numPositions) in enumerate(wheelsToDraw):
                    xPos = windowX + i * self.digitWidth
                    currentOffset = self.wheelOffsets[key][i]
                    targetOffset = targetDigit * self.digitHeight
                    diff = targetOffset - currentOffset
                    totalHeight = self.digitHeight * numPositions
                    if diff > totalHeight / 2:
                        diff -= totalHeight
                    elif diff < -totalHeight / 2:
                        diff += totalHeight
                    if diff != 0:
                        self.animationFrames[key][i] += 1
                        step = self.animationSpeed if diff > 0 else -self.animationSpeed
                        currentOffset += step
                        if abs(currentOffset - targetOffset) < self.animationSpeed or self.animationFrames[key][i] >= self.animationDuration:
                            currentOffset = targetOffset
                            self.animationFrames[key][i] = 0
                        currentOffset = currentOffset % totalHeight
                        if currentOffset < 0:
                            currentOffset += totalHeight
                    else:
                        self.animationFrames[key][i] = 0
                    self.wheelOffsets[key][i] = currentOffset
                    wheelPos = windowY - currentOffset
                    clipRect = pygame.Rect(xPos, windowY, self.digitWidth, self.digitHeight)
                    overlay.set_clip(clipRect)
                    overlay.blit(wheel, (xPos, wheelPos))
                    overlay.blit(wheel, (xPos, wheelPos - totalHeight))
                    overlay.blit(wheel, (xPos, wheelPos + totalHeight))
                    overlay.set_clip(None)
                pygame.draw.rect(overlay, self.textColor, (windowX, windowY, windowWidth, self.digitHeight), 1)
                yPos += spacing
            # STALL indicator for air/space - bottom center
            if self.environment in ["air", "space"] and len(environmentObj.objects) > 0:
                obj = environmentObj.objects[0]
                vel = np.array(obj.velocityVector, dtype=float)
                vel_norm = np.linalg.norm(vel)
                stall = (vel_norm == 0)
                lift_signed = 0.0
                if not stall:
                    orient = np.array(obj.orientationVector, dtype=float)
                    norm_orient = np.linalg.norm(orient)
                    if norm_orient > 0:
                        orient_norm = orient / norm_orient
                        thrust_arr = np.array(obj.thrustForce)
                        force_arr = np.array(getattr(obj, 'forceVector', [0, 0]))
                        aero = force_arr - thrust_arr
                        perp = np.array([-orient_norm[1], orient_norm[0]])
                        lift_signed = np.dot(aero, perp)
                        stall = lift_signed < 0
                stall_x = (self.readoutWidth - 100) // 2
                stall_y = self.readoutHeight - 50 - self.readoutPos[1]
                stall_w = 100
                stall_h = 30
                bg_color = (200, 200, 200) if not stall else (255, 0, 0)
                pygame.draw.rect(overlay, bg_color, (stall_x, stall_y, stall_w, stall_h), border_radius=5)
                pygame.draw.rect(overlay, (0, 0, 0), (stall_x, stall_y, stall_w, stall_h), width=2, border_radius=5)
                text_color = (128, 128, 128) if not stall else (255, 255, 255)
                stall_font = pygame.font.SysFont("courier", 12, bold=True)
                text = stall_font.render("STALL", True, text_color)
                tw, th = text.get_size()
                overlay.blit(text, (stall_x + (stall_w - tw) // 2, stall_y + (stall_h - th) // 2))
        background = pygame.Surface((self.readoutWidth, self.readoutHeight), pygame.SRCALPHA)
        background.fill(self.orientBgColor)
        background.blit(overlay, (0, 0))
        surface.blit(background, self.readoutPos)