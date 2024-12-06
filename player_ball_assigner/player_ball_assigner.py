from utils import *
import numpy as np

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = (int((ball_bbox[0] + ball_bbox[2]) / 2), int((ball_bbox[1] + ball_bbox[3]) / 2))
        
        min_dist = np.inf
        closest_player = -1
        
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_pos)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_pos)
            
            distance = min(distance_left, distance_right)
            
            if distance < self.max_player_ball_distance and distance < min_dist:
                min_dist = distance
                closest_player = player_id
                
        return closest_player