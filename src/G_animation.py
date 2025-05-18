# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that produces an animation from frames to visualize synthetic rock mass
models and other 3D models.
"""

from os import listdir
import cv2
from tqdm import tqdm


def generate_video(image_folder: str, video_name: str,
                   scale: float = 0.6) -> None:
    '''function creates a video from a set of input frames'''
    print('start generating video')
    file_names = []

    for frame_num in range(len(listdir(image_folder))):
        frame_num = str(frame_num)
        if len(frame_num) == 1:
            number = f'0000{frame_num}'
        elif len(frame_num) == 2:
            number = f'000{frame_num}'
        elif len(frame_num) == 3:
            number = f'00{frame_num}'
        file_name = f'Animation_{number}.jpg'
        if file_name in listdir(image_folder):
            file_names.append(file_name)
    print(file_names)
    # setting the frame width, height width of first image
    frame = cv2.imread(fr'{image_folder}\{file_names[0]}')
    height, width, layers = frame.shape
    height, width = int(height*scale), int(width*scale)
    print(height, width)
    video = cv2.VideoWriter(video_name, 0, fps=32,
                            frameSize=(width, height),
                            fourcc=1196444237)

    # Appending the images to the video one by one
    for f_name in tqdm(file_names):
        frame = cv2.imread(fr'{image_folder}\{f_name}')
        frame = cv2.resize(frame, (width, height))
        frame = frame[0:height, 0:width]
        video.write(frame)

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


FOLDER = r'../../../publishing/Parametric_Rockmass_Studies_Paper/02_complexity/figures/animations'
generate_video(image_folder=FOLDER,
               video_name='../old_other/animation.avi')
