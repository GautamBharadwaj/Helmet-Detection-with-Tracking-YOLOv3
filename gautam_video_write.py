import cv2

def save_into_video(path,img):
    video = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = 1200
    frame_height = 675
    out = cv2.VideoWriter('C:/Users/Gautam/Desktop/final_folder/output/output_file2.avi', fourcc, 20.0, (frame_width, frame_height))
    out.write(img)