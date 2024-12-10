import pygame 
from pygame import gfxdraw
import math as m 

def draw_line_as_polygon(gameDisplay, startpos, endpos, width,
                         color, aa=True):  # Wide lines look ugly compared to polygons this draws a polygon as a line
    startx, starty = startpos
    endx, endy = endpos
    angle = m.atan2(endy - starty, endx - startx)
    perpangle = angle - m.pi / 2

    coords = [(startx + m.cos(perpangle) * width, starty + m.sin(perpangle) * width),
              (startx + m.cos(perpangle) * -1 * width, starty + m.sin(perpangle) * -1 * width),
              (endx + m.cos(perpangle) * -1 * width, endy + m.sin(perpangle) * -1 * width),
              (endx + m.cos(perpangle) * width, endy + m.sin(perpangle) * width)]

    pygame.draw.polygon(gameDisplay, color, coords)
    if aa:
        gfxdraw.aapolygon(gameDisplay, coords, color)


def angle_difference(angle1, angle2):
            diff = abs(angle1 - angle2)
            return min(diff, 360 - diff)