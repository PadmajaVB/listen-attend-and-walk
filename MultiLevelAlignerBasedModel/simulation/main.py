# KidsCanCode - Game Development with Pygame video series
# Tile-based game - Part 2
# Collisions and Tilemaps
# Video link: https://youtu.be/ajR4BZBKTr4
import pygame as pg
import sys
from os import path
from settings import *
from sprites import Player, Wall, Path
import time
import copy


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

    def load_data(self):
        game_folder = path.dirname(__file__)
        img_folder = path.join(game_folder, 'img')
        self.map_data = []
        with open(path.join(game_folder, 'Lmap.txt'), 'rt') as f:
            for line in f:
                self.map_data.append(line)
        self.player_img0 = pg.image.load(path.join(img_folder, PLAYER_IMG0)).convert_alpha()
        self.player_img90 = pg.image.load(path.join(img_folder, PLAYER_IMG90)).convert_alpha()
        self.player_img180 = pg.image.load(path.join(img_folder, PLAYER_IMG180)).convert_alpha()
        self.player_img270 = pg.image.load(path.join(img_folder, PLAYER_IMG270)).convert_alpha()

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for row, tiles in enumerate(self.map_data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    colour = BLACK
                    Wall(self, col, row, colour)
                elif tile == '2':
                    colour = YELLOW
                    Path(self, col, row, colour)
                elif tile == '3':
                    colour = PINK
                    Path(self, col, row, colour)
                elif tile == '4':
                    colour = GREEN
                    Path(self, col, row, colour)
                elif tile == '5':
                    colour = BROWN
                    Path(self, col, row, colour)
                elif tile == '6':
                    colour = VIOLET
                    Path(self, col, row, colour)
                elif tile == '7':
                    colour = ORANGE
                    Path(self, col, row, colour)
                elif tile == '8':
                    colour = BLUE
                    Path(self, col, row, colour)
                elif tile == '9':
                    colour = LIGHTGREY
                    Path(self, col, row, colour)
                # elif tile == '.':
                #     colour = WHITE
                #     Path(self, col, row, colour)
                elif tile == 'P':
                    self.player = Player(self, col, row)

    def run(self, path1, actions1):
        # game loop - set self.playing = False to end the game
        self.playing = True
        map_actionsx = {19: 1, 20: 4, 21: 7, 22: 10, 23: 13, 24: 16}
        map_actionsy = {16: 1, 17: 4, 18: 7, 19: 10, 20: 13, 21: 16, 22: 19, 23: 22, 24: 25}
        # path = [[23, 23, 180], [23, 23, 180], [23, 23, 270], [22, 23, 270], [21, 23, 270]]
        # actions = [5, 2, 0, 0, 3]
        # 5: start symbol

        # path = [[21, 21, 90], [21, 21, 90], [22, 21, 90], [23, 21, 90]]
        # actions = [5, 0, 0, 3]

        # path = [[23, 18, 90], [23, 18, 90], [ 23,  18, 180], [ 23,  19, 180], [ 23,  20, 180], [ 23,  21, 180], [ 23,  22, 180], [ 23,  23, 180], [ 23,  23, 270], [ 22,  23, 270], [ 21,  23, 270], [ 21,  23, 180], [21, 23, 90], [21, 23,  0], [21, 22,  0], [21, 21,  0]]
        # actions = [5, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 0, 3]

        path = list()
        actions = list()
        actions.append(5)
        temp = copy.copy(path1[0])
        path.append(temp)
        for x in path1:
            path.append(x)
        for x in actions1:
            actions.append(x)

        # print "Path = ", path
        # print "Actions = ", actions

        # path = [[23, 18, 90], [23, 18, 90], [23, 18, 180], [23, 19, 180], [23, 20, 180], [23, 21, 180], [23, 21, 270], [22, 21, 270], [21, 21, 270]]
        # actions = [5, 2, 0, 0, 0, 2, 0, 0, 3]

        # print "Path = ", path
        # print "Actions = ", actions

        for idx, a in enumerate(path):
            path[idx][0] = map_actionsx[path[idx][0]]
            path[idx][1] = map_actionsy[path[idx][1]]
            # print "Pth --------------- ", path
        # print "Action!!  ", path

        self.player.x = path[0][0]
        self.player.y = path[0][1]
        self.player.orient = path[0][2]
        arg = 0

        if self.playing:
            # self.dt = self.clock.tick(FPS) / 1000
            self.dt = 0
            for idx, act in enumerate(actions):
                if act == 5:
                    orientation = path[idx][2]
                    if orientation == 0:
                        self.player.image = self.player_img0
                    elif orientation == 90:
                        self.player.image = self.player_img90
                    elif orientation == 180:
                        self.player.image = self.player_img180
                    elif orientation == 270:
                        self.player.image = self.player_img270
                    arg = 5
                elif act == 0:
                    coordx, coordy = path[idx + 1][0], path[idx + 1][1]
                    print "x = ", coordx, "  y = ", coordy
                    if coordx > self.player.x:
                        self.player.move(dx=3)
                    elif coordx < self.player.x:
                        print "left"
                        self.player.move(dx=-3)
                    elif coordy > self.player.y:
                        print "Up"
                        self.player.move(dy=3)
                    elif coordy < self.player.y:
                        print "down"
                        self.player.move(dy=-3)
                    arg = 0
                elif act == 1:
                    self.dt = 45
                    rot_speed = PLAYER_ROT_SPEED
                    self.player.rotate(rot_speed)
                    # self.update()
                    arg = 2
                elif act == 2:
                    self.dt = -45
                    rot_speed = PLAYER_ROT_SPEED
                    self.player.rotate(rot_speed)
                    # self.update()
                    arg = 2
                # print "Entering events"
                # self.events(key)
                # print "exited events"
                time.sleep(2)
                self.update(arg)
                self.draw()
                if idx == (len(path) - 1):
                    time.sleep(1)
                    self.quit()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self, arg):
        # update portion of the game loop
        self.all_sprites.update(arg)

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.draw_grid()
        self.all_sprites.draw(self.screen)
        pg.display.flip()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

# create the game object
def simulate(path, actions):
    g = Game()
    g.show_start_screen()
    while True:
        g.new()
        g.run(path, actions)
        g.show_go_screen()
