import xarray
from google.cloud import storage


def setup_gcs_client() -> storage.Client:
    """Sets up and returns a Google Cloud Storage client."""
    return storage.Client.create_anonymous_client()


def load_blob_to_xarray(gcs_bucket: storage.Bucket, blob_name: str) -> xarray.Dataset:
    """Loads a blob from Google Cloud Storage into an xarray Dataset."""
    with gcs_bucket.blob(blob_name).open("rb") as f:
        dataset = xarray.load_dataset(f).compute()
    return dataset


def load_checkpoint(gcs_bucket: storage.Bucket, model_file_name: str):
    """Loads the model checkpoint from Google Cloud Storage."""
    from graphcast import checkpoint, graphcast

    blob = gcs_bucket.blob(model_file_name)
    with blob.open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt


def print_model_info(ckpt):
    """Prints the model description and license."""
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")
