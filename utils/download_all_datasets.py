import os
from pathlib import Path
import io
import zipfile
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"
DATASET_GROUPS = ["machine_hall/", "vicon_room1/", "vicon_room2/"]

SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult"
]

def download_dataset(group, sequence_name, dataset_path):
    download_url = f"{BASE_URL}{group}{sequence_name}/{sequence_name}.zip"
    zip_path = Path(dataset_path) / f"{sequence_name}.zip"
    extract_path = Path(dataset_path) / sequence_name

    print(f"Downloading: {download_url}")

    try:
        with urlopen(download_url) as response:
            total_size = int(response.info().get('Content-Length', -1))
            data = io.BytesIO()

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {sequence_name}") as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    data.write(chunk)
                    pbar.update(len(chunk))

            data = data.getvalue()

            with open(zip_path, 'wb') as f:
                f.write(data)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                print(f"Extracting {sequence_name}...")
                zf.extractall(extract_path)
                print(f"{sequence_name} extracted successfully.")

    except (HTTPError, URLError) as e:
        print(f"Error downloading {sequence_name}: {e}")
    except zipfile.BadZipFile:
        print(f"Invalid ZIP file for {sequence_name}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def download_all_datasets(dataset_path="."):
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    for group in DATASET_GROUPS:
        for sequence in SEQUENCES:
            download_dataset(group, sequence, dataset_path)

if __name__ == "__main__":
    dataset_path = "./EurocMavDatasets"
    download_all_datasets(dataset_path)