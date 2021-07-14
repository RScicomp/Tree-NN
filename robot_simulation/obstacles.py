import pygame

# could add additional feature: [click and drag / click and click again] to create new obstacles
walls = [((0,0),(1000,0)),((0,0),(0,800)),((1000,800),(0,800)),((1000,0),(1000,800))]
obst = [((0,0),(0,0))] #{((100,100),(200,50)),((100,300),(150,360)),((400,600),(350,560))} #,((250,460),(500,700))
for x in walls:
    obst.append(x)


def draw_obstacles(screen):
    for ind,ob in enumerate(obst):
        start = ob[0]
        end = ob[1]
        pygame.draw.line(screen, (ind,ind,ind), start, end, 3)