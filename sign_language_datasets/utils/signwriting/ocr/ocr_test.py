"""ocr tests"""
import os
from unittest import TestCase
from cv2 import cv2

from sign_language_datasets.utils.signwriting.ocr import image_to_fsw


class TestOCR(TestCase):

    def test_should_extract_fsw_from_image(self):
        dirname = os.path.dirname(__file__)
        img_path = os.path.join(dirname, "assets/sign.png")

        img_rgb = cv2.imread(img_path)
        symbols = ['S1f520', 'S1f528', 'S23c04', 'S23c1c', 'S2fb04', 'S2ff00', 'S33b10']

        self.assertEqual(image_to_fsw(img_rgb, symbols), "M239x127S2ff00043x057S23c04071x118S23c1c028x118S1f520062x100S1f528035x100S2fb04054x181S33b10054x083")
