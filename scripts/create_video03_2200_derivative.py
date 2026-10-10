"""Create a bounded Video-03 dataset derivative without mutating raw CVAT XML."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True); p.add_argument("--raw-xml",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--frames",type=int,default=2200)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
    raw_hash=sha256(a.raw_xml); cap=cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise RuntimeError("Cannot open canonical video")
    actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    if actual != a.frames: raise ValueError(f"Expected exactly {a.frames} video frames, found {actual}")
    tree=ET.parse(a.raw_xml); root=tree.getroot(); removed=0; kept=0
    for size in root.findall(".//task/size"): size.text=str(a.frames)
    for track in root.findall("track"):
        for box in list(track.findall("box")):
            if int(box.get("frame","-1")) >= a.frames: track.remove(box); removed+=1
            else: kept+=1
    a.output.mkdir(parents=True)
    derived=a.output/"annotations_0_2199.xml"; tree.write(derived,encoding="utf-8",xml_declaration=True)
    manifest={"status":"COMPLETE","kind":"derived bounded CVAT export; raw XML untouched","frame_range_inclusive":[0,a.frames-1],"frame_count":a.frames,"video":str(a.video.resolve()),"video_sha256":sha256(a.video),"video_metadata":{"fps":fps,"resolution":[width,height]},"raw_annotation":str(a.raw_xml.resolve()),"raw_annotation_sha256":raw_hash,"derived_annotation":derived.name,"derived_annotation_sha256":sha256(derived),"boxes_kept":kept,"boxes_removed_at_or_after_bound":removed,"created_utc":datetime.now(timezone.utc).isoformat()}
    (a.output/"DERIVATIVE_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
