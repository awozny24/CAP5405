# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:21:32 2022

@author: Timothy Lu
"""

import pygame

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    )

#Initialize pygame
pygame.init()

#Define constants for the screen width and height
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

#create screen object

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


#define a player object by extending pygame sprite
#surface drawn on the screen is now an attribute of player
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((75,25))
        self.surf.fill((255, 255,255))
        self.rect = self.surf.get_rect()
# Move the sprite based on kepyress
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -5)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0,5)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-5, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(5, 0)
            
        #Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
       
        



            
            
        
player = Player()

running = True
while running:
    
    #look at every event in the queue
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            #find what type of key
            if event.key == K_ESCAPE:
                running = False
        
        #if event is quit exit the loop
        elif event.type == QUIT:
            running = False
            
    pressed_keys = pygame.key.get_pressed()
            
    player.update(pressed_keys)
    
    #fill the screen with black
    screen.fill((0, 0, 0))
    
    #Draw the player on the screen
    screen.blit(player.surf, player.rect)
    
    pygame.display.flip()
    
