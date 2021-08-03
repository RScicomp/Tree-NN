import pygame

class Baby():

    def __init__(self,image):
        self.image = image
    
    def draw_baby(self, x:int , y: int, screen):
        baby_image = pygame.image.load(self.image)
        baby_image = pygame.transform.scale(baby_image, (100, 100))
        rect = baby_image.get_rect()
        rect.center = (x, y)

        # screen.blit(robot_image, (x, y))
        screen.blit(baby_image, rect)
        self.x = x 
        self.y = y
    
    def emit_shriek(self,screen,circle_radius=12):
        
        colour = (0,0,255)
        circle_x_y = (self.x,self.y)
        border_width = 2
        circle_radius = circle_radius % 1000
        pygame.draw.circle(screen, colour,circle_x_y,circle_radius,border_width)
    
