import pygame
import logging
from main_menu import MainMenu
from shop import Shop
from environment import Air, Space
from renderer import Renderer
from ui_manager import UIManager
import numpy as np

class KSPOS:
    AVAILABLE_ENVIRONMENTS = ["air", "space"]
    
    def __init__(self):
        self.windowWidth = 1000
        self.windowHeight = 800
        self.renderer = Renderer(self.windowWidth, self.windowHeight)
        self.uiManager = UIManager(self.windowWidth, self.windowHeight)
        self.mainMenu = MainMenu(self.windowWidth, self.windowHeight)
        self.shop = None
        self.air = None
        self.space = None
        self.geometry = None
        self.state = "menu"
        self.running = True
        self.clock = pygame.time.Clock()

    def run(self):
        while self.running:
            self.clock.tick(60)  # Cap frame rate at 60 FPS
            self.handleEvents()
            self.update()
            self.render()
        self.renderer.quit()

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.windowWidth, self.windowHeight = event.size
                self.renderer.resize(event.size)
                self.uiManager.resize(event.size)
                self.mainMenu.resize(event.size)
                if self.shop:
                    self.shop.resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.state == "menu":
                    action = self.mainMenu.handleClick(event.pos, self.geometry is not None)
                    if action == "shop":
                        self.state = "shop"
                        self.shop = Shop(self.windowWidth, self.windowHeight)
                    elif action in self.AVAILABLE_ENVIRONMENTS and self.geometry:
                        self.state = action
                        self.initializeEnvironment(action)
                elif self.state == "shop":
                    result = self.shop.handleClick(event.pos)
                    if result == "menu":
                        self.state = "menu"
                        self.shop = None
                    elif result:
                        self.geometry = result
                        self.state = "menu"
                        self.shop = None
                elif self.state in self.AVAILABLE_ENVIRONMENTS:
                    if self.uiManager.uiPanelRect and self.uiManager.uiPanelRect.collidepoint(event.pos):
                        self.uiManager.handleClick(event.pos)
                    else:
                        self.renderer.handleEvent(event, self.uiManager, ui_active=True)
            elif event.type == pygame.KEYDOWN:
                if self.state in self.AVAILABLE_ENVIRONMENTS:
                    if event.key == pygame.K_u:
                        self.uiManager.toggleHiddenUI()
                    elif event.key == pygame.K_ESCAPE:
                        self.state = "menu"
                        if self.state == "air":
                            self.air = None
                        elif self.state == "space":
                            self.space = None
                        self.renderer.firstRender = True  # Reset centering
                    self.renderer.handleEvent(event, self.uiManager, ui_active=True)
                    if self.uiManager.uiPanelRect and self.uiManager.hiddenUIShown:
                        self.uiManager.handleKey(event)
                elif self.state == "shop":
                    self.shop.handleKey(event)

        if self.state in self.AVAILABLE_ENVIRONMENTS:
            keys = pygame.key.get_pressed()
            if self.state == "air":
                self.air.handleKeys(keys)
            elif self.state == "space":
                self.space.handleKeys(keys)
                # Thrust control with arrow keys

            if self.state in self.AVAILABLE_ENVIRONMENTS:
                translation_speed = 5.0
                if keys[pygame.K_d]:
                    self.renderer.globalOffsetX -= translation_speed / self.renderer.scaleX
                if keys[pygame.K_a]:
                    self.renderer.globalOffsetX += translation_speed / self.renderer.scaleX
                if keys[pygame.K_w]:
                    self.renderer.globalOffsetZ -= translation_speed / self.renderer.scaleZ
                if keys[pygame.K_s]:
                    self.renderer.globalOffsetZ += translation_speed / self.renderer.scaleZ
            
    def handleMouseDown(self, event):
        if self.state == "menu":
            action = self.mainMenu.handleClick(event.pos, self.geometry is not None)
            if action == "shop":
                self.state = "shop"
                self.shop = Shop(self.windowWidth, self.windowHeight)
            elif action in self.AVAILABLE_ENVIRONMENTS and self.geometry:
                self.state = action
                self.initializeEnvironment(action)
        elif self.state == "shop":
            self.shop.handleClick(event.pos)

    def handleKeyDown(self, event):
        if self.state in self.AVAILABLE_ENVIRONMENTS:
            if event.key == pygame.K_u:
                self.uiManager.toggleHiddenUI()
            if self.state == "air":
                self.air.handleKey(event)
            elif self.state == "space":
                self.space.handleKey(event)
        elif self.state == "shop":
            self.shop.handleKey(event)

    def initializeEnvironment(self, envType):
        if envType == "air":
            self.air = Air()
            self.air.addObject(self.geometry)
        elif envType == "space":
            self.space = Space()
            self.space.addObject(self.geometry)
        # Reset firstRender to ensure centering
        self.renderer.firstRender = True

    def update(self):
        try:
            if self.state == "air":
                self.air.advanceTime()
                if not self.air.running:
                    logging.info("Air simulation ended, returning to menu")
                    self.air.cleanup()
                    self.air = None
                    self.state = "menu"
            elif self.state == "space":
                self.space.advanceTime()
                if not self.space.running:
                    logging.info("Space simulation ended, returning to menu")
                    self.space.cleanup()
                    self.space = None
                    self.state = "menu"
            elif self.state == "menu":
                self.mainMenu.update()  # Update the main menu for animations
        except Exception as e:
            logging.error(f"Error in update: {e}", exc_info=True)
            self.state = "menu"
            elif self.state == "air":
                self.air = None
            elif self.state == "space":
                self.space = None


    def render(self):
        if self.state == "menu":
            self.mainMenu.render(self.renderer.screen, self.geometry is not None, self.geometry)
            pygame.display.flip()
        elif self.state == "shop":
            self.renderer.renderShop(self.shop)
            self.uiManager.renderShopUI(self.shop)
            pygame.display.flip()
        elif self.state in self.AVAILABLE_ENVIRONMENTS:
            env = self.air if self.state == "air" else self.space if self.state == "space"
            self.renderer.renderEnvironment(env)
            self.uiManager.renderEnvironmentUI(env, self.state)
            pygame.display.flip()

if __name__ == "__main__":
    game = Hydrofoiler()
    game.run()