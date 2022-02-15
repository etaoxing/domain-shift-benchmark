from dsb.dependencies import *


def load_video(video_path, frame_size=None, channel_first=False, grayscale=False):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if frame_size is not None:
        height, width = frame_size

    n_channels = 1 if grayscale else 3
    video = np.zeros((num_frames, height, width, n_channels), dtype=np.uint8)
    assert cap.isOpened()

    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_size is not None:
                frame = cv2.resize(frame, frame_size)

            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame, -1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video[i] = frame
            i += 1
        else:
            break
    cap.release()
    assert i == num_frames

    if channel_first:
        video = video.transpose((0, 3, 1, 2))  # cv2 loads as channel last, convert to channel first
    return video


FOURCC = cv2.VideoWriter_fourcc(*'XVID')


def save_video(video_path, video_frames, fps=20.0, channel_first=False):
    video_dir = '/'.join(video_path.split('/')[:-1])
    os.makedirs(video_dir, exist_ok=True)

    if channel_first:
        video_frames = video_frames.transpose((0, 2, 3, 1))  # convert to channel last

    out = cv2.VideoWriter(video_path, FOURCC, fps, (video_frames.shape[2], video_frames.shape[1]))
    for i in range(len(video_frames)):
        frame = video_frames[i, ...]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def load_image(image_path, channel_first=False, grayscale=False):
    img = cv2.imread(image_path)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel_first:
        img = img.transpose((2, 0, 1))  # cv2 loads as channel last, convert to channel first
    return img


def save_image(image_path, image, channel_first=False):
    image_dir = '/'.join(image_path.split('/')[:-1])
    os.makedirs(image_dir, exist_ok=True)

    if channel_first:
        image = image.transpose((1, 2, 0))  # convert to channel last

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)
