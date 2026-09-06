import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drishti.spatial.floorplan_parser import FloorplanParser

def main():
    p = FloorplanParser()
    res = p.parse_blueprint("seating-plan-for-computer-class.png")
    print(f"Dynamic Blueprint Parser Result: {res['name']}")
    print(f"Total terminals detected: {res['total_terminals']}")
    print("Detected Clusters:")
    for c in res['clusters']:
        print(f"  - {c['name']} ({c['facing']}): {c['terminals']}")

if __name__ == '__main__':
    main()
