"""
Simple color tracking
"""

import argparse
import cv2
import pylint


class ColorTracker:
    def __init__(self):
        pass


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main(arg: argparse.Namespace) -> None:
    print('dupa')


if __name__ == '__main__':
    main(parse_argument())
