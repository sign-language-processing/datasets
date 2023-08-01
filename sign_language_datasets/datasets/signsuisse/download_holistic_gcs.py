import os
from google.cloud import storage
from pose_format import Pose
import tarfile

from tqdm import tqdm


def fix_pose_file(file_path: str):
    with open(file_path, "rb") as buffer:
        pose = Pose.read(buffer.read())
        if len(pose.header.components) == 4:
            return

    print(f'Fixing {file_path}')

    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    with open(file_path, "wb") as f:
        pose.write(f)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')

    fix_pose_file(destination_file_name)


def list_blobs_with_prefix(bucket_name, prefix):
    """Lists all the blobs in the bucket with the given prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    return blobs


def validate_pose_file(file_name: str, buffer: bytes):
    pose = Pose.read(buffer)
    data_frames, data_people, data_points, data_dimensions = pose.body.data.shape
    conf_frames, conf_people, conf_points = pose.body.confidence.shape
    assert data_frames == conf_frames, f'Pose {file_name} has different number of frames in data and confidence'
    assert data_people == conf_people, f'Pose {file_name} has different number of people in data and confidence'
    assert data_points == conf_points, f'Pose {file_name} has different number of points in data and confidence'
    assert data_points == 543, f'Pose {file_name} has different number of points in data ({data_points})'
    assert data_dimensions == 3, f'Pose {file_name} has different number of dimensions in data ({data_dimensions})'


def validate_tar_files(tar_file_name):
    """Iterates over the files in a tar archive and validates each one."""
    with tarfile.open(tar_file_name, "r") as tar:
        for member in tqdm(tar.getmembers(), desc="Validating files"):
            f = tar.extractfile(member)
            if f is not None:
                try:
                    validate_pose_file(member.name, f.read())
                except Exception as e:
                    print(f'Error validating {member.name}')
                    print(e)


def list_files_in_tar_gz(file_path):
    with tarfile.open(file_path, "r") as tar:
        file_names = tar.getnames()
        return [os.path.basename(f) for f in file_names]


def main():
    bucket_name = 'sign-mt-poses'
    prefix = 'external/ss'
    blobs = list_blobs_with_prefix(bucket_name, prefix)

    destination_tar = '/home/nlp/amit/WWW/datasets/poses/holistic/signsuisse.tar'
    existing_files = set(list_files_in_tar_gz(destination_tar))
    print(f'Found {len(existing_files)} valid files in {destination_tar}')
    print(list(existing_files)[:10])

    with tarfile.open(destination_tar, "a") as tar:
        for blob in tqdm(blobs, desc="Reading files"):
            if not blob.name.endswith('/'):
                file_name = os.path.basename(blob.name)
                if file_name not in existing_files:
                    download_blob(bucket_name, blob.name, file_name)
                    tar.add(file_name)
                    os.remove(file_name)

    validate_tar_files(destination_tar)


if __name__ == "__main__":
    main()

# Extract: tar xf signsuisse.tar --directory signsuisse
# Compress: tar -cvf signsuisse.tar -C signsuisse .
# import os
#
# from pose_format import Pose
#
# directory = '/home/nlp/amit/WWW/datasets/poses/holistic/signsuisse'
# for filename in tqdm(os.listdir(directory)):
#     if filename.endswith(".pose"):
#         file_path = os.path.join(directory, filename)
#
#         fix_pose_file(file_path)
