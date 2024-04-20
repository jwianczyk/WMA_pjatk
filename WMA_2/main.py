#pylint: disable=no-member
"""
Simple color tracking
"""

import argparse
import cv2
from PIL import Image
import numpy as np
from enum import Enum


class ProcessingType(Enum):
    """
    Class for types of processing enum
    """
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5


class PreprocessingType(Enum):
    """
    Class for erosion and dilatation enum
    """
    ERODE = 0
    DILATE = 1


class ColorTracker:
    """
    Controller for processing frames.
    """

    _COLOR_RELIANT_PROCESSES = [
        ProcessingType.MASK,
        ProcessingType.TRACKER
    ]

    def __init__(self, video_path: str,
                 hue_range: int, saturation_range: int, value_range: int) -> None:
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise ValueError(f'Unable to open video at path {video_path}')
        self._tracked_color: None or tuple[int, int, int] = None
        self._frame: None or np.ndarray = None
        self._processed_frame: None or np.ndarray = None
        self._processing_type: ProcessingType = ProcessingType.RAW
        self._erosion: int = 0
        self._dilation: int = 0

        self._ranges_dictionary = {
            0: hue_range,
            1: saturation_range,
            2: value_range
        }

        self._processing_handler = {
            ProcessingType.RAW: self._get_current_frame,
            ProcessingType.TRACKER: self._prepare_tracking_layer,
            ProcessingType.HUE: self._prepare_hue_map,
            ProcessingType.SATURATION: self._prepare_saturation_map,
            ProcessingType.VALUE: self._prepare_value_map,
            ProcessingType.MASK: self._prepare_merged_mask,
            PreprocessingType.ERODE: self._add_erosion,
            PreprocessingType.DILATE: self._add_dilation
        }

    def set_processing_type(self, ptype: ProcessingType) -> None:
        """
        Setter for _processing_type.
        :param ptype: processing type
        :return: None
        """
        self._processing_type = ptype

    def set_reference_color_by_position(self, x: int, y: int) -> None:
        """
        Setter for _tracked_color.
        :param x: X coordinate of cursor
        :param y: Y coordinate of cursor
        :return:
        """
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y, x, :]

    def update_ed_depths(self, preprocess: PreprocessingType) -> None:
        """
        Handles preprocessing changes.
        :param preprocess: Preprocessing type
        :return: None
        """
        self._processing_handler[preprocess]()

    def update_frame(self) -> bool:
        """
        Method for updating video.
        :return: bool: True if video read successfully
        """
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._preprocess_frame()
            self._process_frame()
        return read_successful

    def _add_erosion(self) -> None:
        """
        Increase erosion counter
        :return: None
        """
        self._erosion += 1

    def _add_dilation(self) -> None:
        """
        Increase dilatation counter
        :return: None
        """
        self._dilation += 1

    def _erode(self, img: np.ndarray) -> np.ndarray:
        """
        Method for applying erosion
        :param img: image to erode
        :return: np.ndarray eroded image
        """
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(img, kernel, iterations=self._erosion)

    def _dilate(self, img: np.ndarray) -> np.ndarray:
        """
        Method for applying dilatation
        :param img: image to dilatate
        :return: np.ndarray dilatated image
        """
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(img, kernel, iterations=self._dilation)

    def _get_current_frame(self) -> np.ndarray:
        """
        Getter for current frame
        :return: np.ndarray: current frame
        """
        return self._frame

    def _get_mask(self, mask: np.ndarray, index: int) -> np.ndarray:
        """
        Method for preparing value detection in layer
        :param mask:
        :param index: index of layer
        :return: np.ndarray: black mask with white color in place of tracked color
        """
        related_tolerance = self._ranges_dictionary[index]
        left_inner = np.where(mask <= self._tracked_color[index] + related_tolerance, mask, 0)
        return np.where(self._tracked_color[index] - related_tolerance <= left_inner, 255, 0)

    def _get_tracked_color_frame(self, mask: np.ndarray) -> np.ndarray:
        """
        Getter for tracking color with rectangle
        :param mask: mask containing merged h,s and v masks
        :return: frame with rectangle around the tracked color
        """
        mask_image = Image.fromarray(mask)
        bounding_box = mask_image.getbbox()
        if bounding_box is None:
            return mask
        x1, y1, x2, y2 = bounding_box
        drawing = cv2.rectangle(np.zeros_like(mask, dtype=np.uint8),
                                (x1, y1), (x2, y2), 255, 2)
        return drawing

    def _prepare_hue_map(self) -> np.ndarray:
        """
        Method for getting hue layer
        :return: np.ndarray: hue layer
        """
        return cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)[:, :, 0]

    def _prepare_saturation_map(self) -> np.ndarray:
        """
        Method for getting saturation layer
        :return: np.ndarray: saturation layer
        """
        return cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)[:, :, 1]

    def _prepare_value_map(self) -> np.ndarray:
        """
        Method for getting value layer
        :return: np.ndarray: value layer
        """
        return cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)[:, :, 2]

    def _validate_color(self) -> None:
        if self._tracked_color is None:
            raise ValueError('No color is tracked right now.')

    def _prepare_merged_mask(self) -> np.ndarray:
        """
        Method for creating color mask layer
        :return: color mask consisting of h, s and v
        """
        self._validate_color()
        mask_hue = self._get_mask(self._prepare_hue_map(), 0)
        mask_saturation = self._get_mask(self._prepare_saturation_map(), 1)
        mask_value = self._get_mask(self._prepare_value_map(), 2)
        return (mask_hue & mask_saturation & mask_value).astype(np.uint8)

    def _prepare_tracking_layer(self) -> np.ndarray:
        """
        Method for creating color mask tracking layer
        :return: color mask consisting of h, s and v
        """
        return self._get_tracked_color_frame(self._prepare_merged_mask())

    def _preprocess_frame(self) -> None:
        """
        Method for preprocessing frame
        :return: None
        """
        eroded = self._erode(self._frame)
        eroded_and_dilated = self._dilate(eroded)
        self._frame = eroded_and_dilated

    def _process_frame(self) -> None:
        """
        Method for processing frame
        :return: None
        """
        self._processed_frame = self._processing_handler[self._processing_type]()

    def get_processed_frame(self) -> np.ndarray:
        """
        Method for altering current frame with processing type
        :return: None
        """
        processed_frame_copy = self._processed_frame.copy()
        return processed_frame_copy


class Display:
    """
    Method acting as view
    """
    def __init__(self, window_name: str) -> None:
        self._window = cv2.namedWindow(window_name)
        self._window_name = window_name

    def update_display(self, image: np.ndarray) -> None:
        """
        Method for updating the video with provided frame
        :param image: np.ndarray: frame
        :return: None
        """
        cv2.imshow(self._window_name, image)

    def get_window_name(self) -> str:
        """
        Getter for window name
        :return: str: window name
        """
        return self._window_name


class EventHandler:
    """
    Class for handling events between model and view
    """
    PROCESSING_TYPE_KEYMAP = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    PREPROCESSING_TYPE_KEYMAP = {
        ord('e'): PreprocessingType.ERODE,
        ord('d'): PreprocessingType.DILATE
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int) -> None:
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def _handle_mouse(self, event, x, y, flags=None, params=None) -> None:
        """
        Method for handling mouse inputs
        :param event: event
        :param x: x position of cursor
        :param y: y position of cursor
        :return: None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color_by_position(x, y)

    def _handle_keys(self) -> bool:
        """
        Method for handling inputs from keyboard
        :return: True if the key is pressed
        """
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        elif keycode in EventHandler.PROCESSING_TYPE_KEYMAP.keys():
            self._tracker.set_processing_type(EventHandler.PROCESSING_TYPE_KEYMAP[keycode])
        elif keycode in EventHandler.PREPROCESSING_TYPE_KEYMAP.keys():
            self._tracker.update_ed_depths(self.PREPROCESSING_TYPE_KEYMAP[keycode])
        return True

    def handle_events(self) -> bool:
        """
        Public method for handling inputs from keyboard
        :return: method corresponding to keyboard input
        """
        return self._handle_keys()


def parse_argument() -> argparse.Namespace:
    """
    Method for parsing arguments
    :return: argparse with parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', type=str,
                        required=True, help='Path to the video file')
    parser.add_argument('--hue-range', type=str, default=5, required=False,
                        help='Range of permissible hue')
    parser.add_argument('--saturation-range', type=str, default=50, required=False,
                        help='Range of permissible saturation')
    parser.add_argument('--value-range', type=str, default=50, required=False,
                        help='Range of permissible value')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function
    :param args: parsed arguments
    :return: None
    """
    try:
        window_name = 'Color Tracker'
        waitkey_timeout = 10
        tracker = ColorTracker(
            args.video_path, args.hue_range, args.saturation_range, args.value_range)
        display = Display(window_name)
        event_handler = EventHandler(tracker, display, waitkey_timeout)
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
