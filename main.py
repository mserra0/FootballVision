from utils import *
from trackers import *
from team_assigner import *
from player_ball_assigner import *
from camera_movement import *
from view_transformation import *
from speed_and_distance import *
import numpy as np
import cv2

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def main():
    #frames = read_video('video1.1.mp4')
    frames = [cv2.imread('image3.jpg')]

    tracker = Tracker('models/yolov5lu/best.pt')

    tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    camera_movement = CameraMovement(frames[0])
    frames_camera_movement = camera_movement.get_camera_movement(frames)#, read_from_stub=False, stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement.adjust__tracks_positions(tracks, frames_camera_movement)
    
    tracks["ball"] = tracker.ball_interpolation(tracks["ball"])
    
    speed_dist = SpeedAndDistance_Estimator()
    #speed_dist.add_speed_and_distance(tracks)
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])
    
    for n_frame, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[n_frame], track["bbox"], player_id)
            tracks["players"][n_frame][player_id]["team"] = team
            tracks["players"][n_frame][player_id]["team_color"] = team_assigner.team_colors[team]
            
    player_assigner = PlayerBallAssigner()
    team_ball_possession = []
    
    for n_frame, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][n_frame][1]["bbox"]
        closest_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if closest_player != -1:
            tracks["players"][n_frame][closest_player]["ball_possession"] = True
            team_ball_possession.append(tracks["players"][n_frame][closest_player]["team"])
        else:
            team_ball_possession.append(team_ball_possession[-1])
        
    team_ball_possession = np.array(team_ball_possession)
    
    output = tracker.draw_annotations(frames, tracks, team_ball_possession)
    
    output = camera_movement.draw_camera_movement(output, frames_camera_movement)
    
    #speed_dist.draw_speed_and_distance(output, tracks)
    
    save_video(output, 'frames_video/output_video.avi')


if __name__ == '__main__':
    main()
    