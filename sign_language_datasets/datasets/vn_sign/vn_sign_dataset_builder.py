import numpy as np
import tensorflow_datasets as tfds
import os
from sklearn.model_selection import train_test_split
import logging

class VnSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Vietnamese Sign Language (VnSign) dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Manually download the Vietnamese sign language videos and place them in the `manual/vn_sign` directory.
    """

    def _info(self):
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=("This is a dataset for Vietnamese sign language."),
            features=tfds.features.FeaturesDict({
                'video': tfds.features.Video(shape=(None, None, None, 3), dtype=np.uint8),
                'label': tfds.features.Text(),
            }),
            supervised_keys=('video', 'label'),
            homepage='https://tudienngonngukyhieu.com',
            citation=r"""@article{vietnamese-sign-language-2021,
                            author = {Nguyen Quoc Khanh},
                            title = {Vietnamese Sign Language},
                            journal = {Journal of Datasets},
                            year = {2024},
                        }""",
        )

    def _split_generators(self, dl_manager):
        """Define the dataset splits."""
        manual_dir = dl_manager.manual_dir / 'vn_sign'
        
        # Ensure manual_dir exists and is not empty
        if not manual_dir.exists() or not any(manual_dir.iterdir()):
            raise FileNotFoundError(f"Manual directory {manual_dir} is empty. Please manually download the videos and place them in the directory.")
        
        video_paths = list(manual_dir.glob('*.mp4'))
        logging.info(f"Found {len(video_paths)} video files in {manual_dir}")
        for path in video_paths:
            logging.info(f"Video file found: {path}")
        
        if len(video_paths) == 0:
            raise ValueError(f"No video files found in {manual_dir}. Ensure you have .mp4 files in this directory.")
        
        train_paths, test_paths = train_test_split(video_paths, test_size=0.2, random_state=42)
        
        return {
            'train': self._generate_examples(train_paths),
            'test': self._generate_examples(test_paths),
        }

    def _generate_examples(self, paths):
        """Generator of examples for each split."""
        logging.info(f"Generating examples from: {paths}")
        count = 0
        for video_path in paths:
            count += 1
            yield video_path.stem, {
                'video': video_path,
                'label': video_path.stem,
            }
        logging.info(f"Number of examples generated: {count}")

if __name__ == "__main__":
    output_dir = 'C:/Users/Admin/tensorflow_datasets/downloads/manual/vn_sign'
    print(f"Data should be manually placed in: {output_dir}")
