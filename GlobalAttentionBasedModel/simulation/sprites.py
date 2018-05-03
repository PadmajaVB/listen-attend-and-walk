import pygame as pg
from settings import *
import time


class Player(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image = game.player_img0
        # self.image.fill(DARKGREEN)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rot = 0
        self.rot_speed = PLAYER_ROT_SPEED

    def move(self, dx=0, dy=0):
        if not self.collide_with_walls(dx, dy):
            self.x += dx
            self.y += dy

    def collide_with_walls(self, dx=0, dy=0):
        for wall in self.game.walls:
            if wall.x == self.x + dx and wall.y == self.y + dy:
                return True
        return False

    def rotate(self, rot_speed):
        print "Entered rotate"
        self.rot_speed = rot_speed

    def update(self, arg):
        if arg == 0 or arg == 5:
            print "Update"
            self.rect.x = self.x * TILESIZE
            self.rect.y = self.y * TILESIZE
        elif arg == 2:
            self.rect.x = self.x * TILESIZE
            self.rect.y = self.y * TILESIZE
            print "self.dt = ", self.game.dt
            self.rot = (self.rot_speed * self.game.dt) % 360
            print "Rotation = ", self.rot
            self.image = pg.transform.rotate(self.image, self.rot)


class Wall(pg.sprite.Sprite):
    def __init__(self, game, x, y, colour):
        self.groups = game.all_sprites, game.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Path(pg.sprite.Sprite):
    def __init__(self, game, x, y, colour):
        self.groups = game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE