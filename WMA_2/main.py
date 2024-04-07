"""
Simple color tracking
"""

import pylint
import argparse
import cv2
import numpy as np
from enum import Enum
import random as rng


class ProcessingType(Enum):
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5

# Model
class ColorTracker:
    def __init__(self, video_path: str, tracked_color: None or tuple[int, int, int]) -> None:
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise ValueError(f'Unable to open video at path {video_path}')
        self._tracked_color: None or tuple[int, int, int] = tracked_color
        self._frame: None or np.ndarray = None
        self._processed_frame: None or np.ndarray = None
        self._processing_type: ProcessingType = ProcessingType.RAW
        self.erode: int = 0
        self.dilate: int = 0

    def set_processing_type(self, ptype: ProcessingType) -> None:
        self._processing_type = ptype

    def set_reference_color_by_position(self, x: int, y: int) -> None:
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y, x, :]

    def update_frame(self) -> bool:
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._process_frame()
        return read_successful

    def _process_frame(self) -> None:
        if self._processing_type == ProcessingType.RAW:
            self._processed_frame = self._frame
            return
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        hue = hsv_frame[:, :, 0]
        saturation = hsv_frame[:, :, 1]
        value = hsv_frame[:, :, 2]
        if self._processing_type == ProcessingType.HUE:
            self._processed_frame = hue
            return
        elif self._processing_type == ProcessingType.SATURATION:
            self._processed_frame = saturation
            return
        elif self._processing_type == ProcessingType.VALUE:
            self._processed_frame = value
            return
        if self._tracked_color is None:
            raise ValueError(f'Attempted processing mode that requires a tracking color set without it set.')
        mask = np.zeros_like(hue)
        mask[hue == self._tracked_color[0]] = 255
        mask[saturation == self._tracked_color[1]] = 255
        mask[value == self._tracked_color[2]] = 255
        if self._processing_type == ProcessingType.MASK:
            self._processed_frame = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

        drawing = np.zeros_like(mask, dtype=np.uint8)

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0] + boundRect[i][2]),
                           int(boundRect[i][1] + boundRect[i][3])), color, 2)
        if self._processing_type == ProcessingType.TRACKER:
            self._processed_frame = drawing

    def get_frame(self) -> np.ndarray:
        if self._frame is None:
            raise ValueError('Attempted to get frame from uninitialized color tracker')
        return self._frame.copy()

    def get_processed_frame(self) -> np.ndarray:
        processed_frame_copy = self._processed_frame.copy()
        kernel = np.ones((2, 2), np.uint8)
        if self.erode != 0:
            processed_frame_copy = cv2.erode(self._processed_frame.copy(), kernel, iterations=self.erode)
        if self.dilate != 0:
            processed_frame_copy = cv2.dilate(self._processed_frame.copy(), kernel, iterations=self.dilate)
        if self._processing_type == ProcessingType.RAW:
            return self._processed_frame.copy()
        return processed_frame_copy


# View
class Display:
    def __init__(self, window_name: str) -> None:
        self._window = cv2.namedWindow(window_name)
        self._window_name = window_name

    def update_display(self, image: np.ndarray) -> None:
        cv2.imshow(self._window_name, image)

    def get_window_name(self) -> str:
        return self._window_name


# Controller
class EventHandler:
    PROCESSING_TYPE_KEYMAP = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int) -> None:
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def _handle_mouse(self, event, x, y, flags, params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color_by_position(x, y)
            print(self._tracker._tracked_color)

    def _handle_keys(self) -> bool:
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        elif keycode in EventHandler.PROCESSING_TYPE_KEYMAP.keys():
            self._tracker.set_processing_type(EventHandler.PROCESSING_TYPE_KEYMAP[keycode])
        elif keycode == ord('e'):
            self._tracker.erode += 1
            print(f'Current erode: {self._tracker.erode}')
        elif keycode == ord('d'):
            self._tracker.dilate += 1
            print(f'Current dilation: {self._tracker.dilate}')
        return True

    def handle_events(self) -> bool:
        return self._handle_keys()


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', type=str,
                        required=True, help='Path to the video file')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    try:
        WINDOW_NAME = 'Color Tracker'
        WAITKEY_TIMEOUT = 10
        tracker = ColorTracker(args.video_path, (23, 43, 12))
        display = Display(WINDOW_NAME)
        event_handler = EventHandler(tracker, display, WAITKEY_TIMEOUT)
        while True:
            if not tracker.update_frame():
                break
            display.update_display(tracker.get_processed_frame())
            if not event_handler.handle_events():
                break

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main(parse_argument())
