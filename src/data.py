import os

import cv2
import numpy as np
from torch.utils.data import Dataset


def get_images_annotations(data_path) -> list[tuple[np.ndarray, float, float]]:
    """
    Retrieves images and annotations from csv files

    :param data_path: path to directory with csv files
    :return: list of tuples (image, forward_signal, left_signal)
    """
    images_annotations = []
    for filename in os.listdir(data_path):
        csv_filename = os.path.join(data_path, filename)

        if os.path.isfile(csv_filename):
            img_dir = ".".join(csv_filename.split(".")[:-1])

            with open(csv_filename, "r") as csv_file:
                for line in csv_file:
                    img_no, forward_signal, left_signal = line.split(",")
                    forward_signal = float(forward_signal)
                    left_signal = float(left_signal)

                    img_path = img_dir + "/" + str(img_no).zfill(4) + ".jpg"
                    image = cv2.imread(img_path)
                    image_reshaped = np.transpose(image, (2, 0, 1))
                    images_annotations.append(
                        (image_reshaped, forward_signal, left_signal)
                    )
    return images_annotations


class RobotDataset(Dataset):
    """
    Dataset for robot route images
    """

    def __init__(self, images_annotations, transform: bool = True):
        images, forward_signals, left_signals = list(zip(*images_annotations))
        self.images = images
        self.forward_signals = forward_signals
        self.left_signals = left_signals
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def transform(self, image):
        # TODO: preprocess image
        return image

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        forward_signal = np.clip(self.forward_signals[idx], -1, 1)
        left_signal = np.clip(self.left_signals[idx], -1, 1)
        return image, forward_signal, left_signal
