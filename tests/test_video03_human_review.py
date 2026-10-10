import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from build_video03_human_review import enrich_proposals, temporal_clusters, diagnose
from serve_video03_human_review import validate_decision, ReviewStore, handler_for
from probe_video03_phone_misses import coverage


def proposal(frame, box=(10,10,30,30), ident=None):
    return {'frame_index':frame,'bbox_xyxy_px':list(box),'proposal_id':f'p{frame}_{box[0]}','associated_entity_id':ident}


def test_temporal_grouping_keeps_all_rows_and_does_not_bridge_long_gaps():
    rows=[proposal(1),proposal(2,ident=8),proposal(12),proposal(23),proposal(23,(100,100,120,120))]
    clusters=temporal_clusters(rows)
    assert sorted(len(c) for c in clusters)==[1,1,3]
    assert sum(map(len,clusters))==len(rows)
    assert all(len({r['frame_index'] for r in c})==len(c) for c in clusters)


def test_same_frame_objects_are_not_merged_or_spatially_conflated():
    clusters=temporal_clusters([proposal(1),proposal(1,(100,100,120,120)),proposal(2),proposal(2,(100,100,120,120))])
    assert len(clusters)==2
    assert all(len(c)==2 for c in clusters)


def test_diagnostic_crop_coverage_never_implies_a_phone_label():
    assert coverage([10,10,30,30],[0,0,40,40])==1
    assert coverage([10,10,30,30],[0,0,20,40])==.5
    assert coverage([10,10,30,30],[40,40,60,60])==0


def test_provenance_exact_join_restores_association_and_unknown_status():
    meta={'model_weight':'phone.pt','crop_xyxy':[1,2,3,4]}
    obj={'object_class':'MOBILE_PHONE','bbox_xyxy_px':[10,10,30,30],'confidence':.7,
        'metadata':meta,'associated_entity_id':9,'associated_seat_id':None,'association_confidence':.3}
    frames={1:{'objects':[obj],'model_provenance':{'phone':{'path':'phone.pt','sha256':'hash'}}}}
    exported={'frame_index':1,'bbox_xyxy_px':[10,10,30,30],'raw_confidence':.7,'inference_metadata':meta,'review_status':'NEEDS_REVIEW'}
    a=enrich_proposals([exported],frames,'exporthash')[0]
    b=enrich_proposals([exported],frames,'exporthash')[0]
    assert a['proposal_id']==b['proposal_id']
    assert a['review_status']=='UNKNOWN'
    assert a['associated_entity_id']==9
    assert a['model_sha256']=='hash'
    assert exported['review_status']=='NEEDS_REVIEW'
    frames[1]['objects'].append(obj)
    with pytest.raises(ValueError,match='Non-unique'):
        enrich_proposals([exported],frames,'exporthash')


def test_missing_geometry_is_not_automatically_an_occlusion_or_annotation_error():
    label={'frame_index':1,'track_id':0,'class':'Student','outside':False,'geometry_valid':True,
        'bbox_xyxy_px':[10,10,30,30],'attributes':{'visibility':'clear'},'occluded':False}
    failures,matches=diagnose([label],{1:{'people':[],'objects':[]}}, {})
    assert matches==[]
    assert failures[0]['diagnostic_category']=='NO_OVERLAPPING_RETAINED_DETECTION'
    assert failures[0]['confirmed_cause'] is None
    assert not failures[0]['annotation_change_applied']


def test_review_status_never_defaults_to_not_phone():
    proposals={'p1':{'segment_id':'s'},'p2':{'segment_id':'s'}}
    base={'proposal_ids':['p1'],'reviewer':'human','status':'NOT_PHONE'}
    with pytest.raises(ValueError,match='acknowledgement'):
        validate_decision(base,proposals)
    assert validate_decision({**base,'manual_ack':True,'status':'UNKNOWN'},proposals)['status']=='UNKNOWN'
    with pytest.raises(ValueError,match='Every selected'):
        validate_decision({**base,'manual_ack':True,'proposal_ids':['p1','p2']},proposals)
    with pytest.raises(ValueError,match='Unsupported status'):
        validate_decision({**base,'manual_ack':True,'status':'FALSE_POSITIVE'},proposals)


def make_store(tmp_path):
    from hashlib import sha256
    data={'proposals':[{'proposal_id':'p1','segment_id':'s'},{'proposal_id':'p2','segment_id':'s'}]}
    p=tmp_path/'review_data.json'
    p.write_text(json.dumps(data))
    (tmp_path/'READY_FOR_HUMAN_REVIEW.json').write_text(json.dumps({'review_data_sha256':sha256(p.read_bytes()).hexdigest()}))
    return ReviewStore(tmp_path)


def test_decisions_are_append_only_and_reloaded_with_provenance(tmp_path):
    store=make_store(tmp_path)
    assert set(store.state()['statuses'].values())=={'UNKNOWN'}
    payload={'proposal_ids':['p1'],'status':'NOT_PHONE','reviewer':'test reviewer','manual_ack':True}
    store.decide(payload)
    store.decide({**payload,'status':'AMBIGUOUS'})
    assert len(store.path.read_text().splitlines())==2
    reloaded=ReviewStore(tmp_path)
    assert reloaded.state()['statuses']=={'p1':'AMBIGUOUS','p2':'UNKNOWN'}
    assert not reloaded.state()['training_authorized']
    (tmp_path/'review_data.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        ReviewStore(tmp_path)


def test_local_http_workflow_and_video_byte_ranges(tmp_path):
    store=make_store(tmp_path)
    (tmp_path/'preview.mp4').write_bytes(b'0123456789')
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(store,'token'))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/state') as response:
            assert json.load(response)['statuses']['p1']=='UNKNOWN'
        request=Request(base+'/preview.mp4',headers={'Range':'bytes=2-5'})
        with urlopen(request) as response:
            assert response.status==206
            assert response.read()==b'2345'
        payload=json.dumps({'proposal_ids':['p1'],'status':'CONFIRMED_PHONE','reviewer':'test reviewer','manual_ack':True}).encode()
        bad=Request(base+'/api/decision',data=payload,headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(bad)
        assert error.value.code==403
        assert not store.path.exists()
        good=Request(base+'/api/decision',data=payload,headers={'Content-Type':'application/json','Origin':base,'X-Review-Token':'token'})
        with urlopen(good) as response:
            assert json.load(response)['decision']['status']=='CONFIRMED_PHONE'
        assert store.state()['statuses']['p2']=='UNKNOWN'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
