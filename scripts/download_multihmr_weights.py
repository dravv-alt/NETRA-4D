import os
import urllib.request
import sys

target_dir = os.path.abspath("models/multiHMR")
os.makedirs(target_dir, exist_ok=True)

# We download multiHMR_672_S (Fast, lightweight ~130MB, designed to fit 4-6GB GPU easily)
# And multiHMR_672_B if needed
models = [
    ("multiHMR_672_S.pt", "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_S.pt"),
]

for filename, url in models:
    dest = os.path.join(target_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
        print(f"{filename} already exists at {dest}")
        continue
    print(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Successfully saved to {dest} ({os.path.getsize(dest) / 1e6:.2f} MB)")
