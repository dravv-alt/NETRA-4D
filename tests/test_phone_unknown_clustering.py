from pathlib import Path
import sys
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'; sys.path.insert(0,str(SCRIPTS))
from cluster_phone_unknowns import near_window
from build_video03_human_review import temporal_clusters

def row(frame, box=(0,0,20,20)):
 return {'frame_index':frame,'bbox_xyxy_px':list(box),'proposal_id':f'{frame}-{box[0]}','associated_entity_id':1}

def test_phase1_clustering_is_deterministic_and_does_not_merge_separate_objects():
 rows=[row(1),row(2),row(3),row(30),row(30,(100,100,120,120)),row(31,(100,100,120,120))]
 a=[[r['proposal_id'] for r in x] for x in temporal_clusters(rows,10)]
 b=[[r['proposal_id'] for r in x] for x in temporal_clusters(rows,10)]
 assert a==b and sorted(map(len,a))==[1,2,3]

def test_known_window_priority_overlap_is_explicit():
 assert near_window(440,445,25)
 assert not near_window(800,810,25)
