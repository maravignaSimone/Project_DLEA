# define RGB colors for each class
# with help of https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# labels by the author of the dataset

class_mapping={
    (0, 0, 0): 0, # unlabelled
    (111, 74,  0): 1, # static
    ( 81,  0, 81): 2, # ground
    (128, 64,127): 3, # road
    (244, 35,232): 4, # sidewalk
    (250,170,160): 5, # parking
    (230,150,140): 6, # rail track
    (70, 70, 70): 7, # building
    (102,102,156): 8, # wall
    (190,153,153): 9 , # fence
    (180,165,180): 11, # guard rail
    (150,100,100): 12, # bridge
    (150,120, 90): 13, # tunnel
    (153,153,153): 14, # pole
    (153,153,153): 15, # polegroup
    (250,170, 30): 16, # traffic light
    (220,220,  0): 17, # traffic sign
    (107,142, 35): 18, # vegetation
    (152,251,152): 19, # terrain
    ( 70,130,180): 20, # sky
    (220, 20, 60): 21, # person
    (255,  0,  0): 22, # rider
    (  0,  0,142): 23, # car
    (  0,  0, 70): 24, # truck
    (  0, 60,100): 25, # bus
    (  0,  0, 90): 26, # caravan
    (  0,  0,110): 27, # trailer
    (  0, 80,100): 28, # train
    (  0,  0,230): 29, # motorcycle
    (119, 11, 32): 30, # bicycle
}