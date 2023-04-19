import os
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')

def list_blobs_with_prefix(bucket_name, prefix):
    """Lists all the blobs in the bucket with the given prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    return blobs

def main():
    bucket_name = 'sign-mt-poses'
    prefix = 'external/ss'
    destination_directory = 'holistic'

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    blobs = list_blobs_with_prefix(bucket_name, prefix)
    for blob in blobs:
        if not blob.name.endswith('/'):
            file_name = os.path.basename(blob.name)
            destination_path = os.path.join(destination_directory, file_name)
            if not os.path.exists(destination_path):
                download_blob(bucket_name, blob.name, destination_path)

if __name__ == "__main__":
    main()
