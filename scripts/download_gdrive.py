import os
import sys
import gdown

files = [
    ("12AuvNrLQun7DLr3Oxq60WC6bLeZNWeQI", "dataset_part1"),
    ("1Dk-t3uNEZdfiNB0X5H6THeMs1PKhW53V", "dataset_part2"),
    ("1MFD3E5hu-Jj3H1cPBSMqOaFE7W1mJB7y", "dataset_part3"),
    ("1nbzx0eA4qvRMjoPzwuukqNubqFSDtxhe", "dataset_part4"),
    ("1ynPcGguy9fi0kaw9AHfmvAzwpvxUX1Vc", "dataset_part5"),
]

output_dir = os.path.abspath("dataset_gdrive")
os.makedirs(output_dir, exist_ok=True)

for file_id, name in files:
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"\n--- Downloading {name} (ID: {file_id}) ---")
    try:
        # gdown format without output filename keeps remote filename
        downloaded = gdown.download(url=url, output=output_dir + "/", quiet=False)
        print(f"Downloaded: {downloaded}")
    except Exception as e:
        print(f"Error downloading {file_id}: {e}")
