import torch 
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    record_buffer = torch.load(args.input_file)
    video = cv2.VideoCapture(record_buffer["rgb"])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    video_path = args.output_file
    size = (width, height)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    i = 0
    while True:
        i += 1
        ret, frame, = video.read()
        if not ret:
            break

        if i >= len(record_buffer["fov_loc"]):
            print(i)
            continue
        y_loc, x_loc = record_buffer["fov_loc"][i]
        fov_size = record_buffer["fov_size"]

        # The fov_loc is set within a 84x84 grid while the video output is 256x256
        # To scale them accordingly we multiply with the following
        COORD_SCALING = 256 / 84
        x_loc = int(x_loc * COORD_SCALING)
        y_loc = int(y_loc * COORD_SCALING)
        fov_size = (int(fov_size[0] * COORD_SCALING), int(fov_size[1] * COORD_SCALING))

        top_left = (x_loc, y_loc)
        bottom_right = (x_loc + fov_size[0], y_loc + fov_size[1])
        color = (255, 0, 0)
        thickness = 1
        frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        # frame[x_loc, y_loc] = [255, 0, 0]
        video_writer.write(frame)

    video.release()
    video_writer.release()

