"""Camera-independent context crops from detector boxes, never annotation locations."""
import math


def _iou(a,b):
    overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    area = lambda p: max(0,p[2]-p[0]) * max(0,p[3]-p[1])
    return overlap/max(1e-9,area(a)+area(b)-overlap)


def person_context_crops(boxes, width, height, size=384):
    if size <= 0 or width <= 0 or height <= 0:
        raise ValueError('Crop and frame dimensions must be positive')
    regions = []
    for box in boxes:
        x1,y1,x2,y2 = map(float,box)
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
            continue
        side = min(max(size, round(max(x2-x1,y2-y1)*1.25)),width,height)
        cx = max(0,min(width-side,round((x1+x2)/2-side/2)))
        cy = max(0,min(height-side,round((y1+y2)/2-side/2)))
        region = (cx,cy,cx+side,cy+side)
        if region not in regions:
            regions.append(region)
    return regions


class PersonCropMemory:
    """Short-lived detector ROI proposals. Never retains phone detections."""
    def __init__(self, hold_seconds=1.):
        if hold_seconds < 0 or not math.isfinite(hold_seconds):
            raise ValueError('ROI hold time must be finite and nonnegative')
        self.hold_seconds = hold_seconds
        self.tracks = {}
        self.last_timestamp = None
        self.frame_size = None

    def regions(self, people, width, height, timestamp, size=384):
        if self.frame_size != (width,height) or self.last_timestamp is not None and timestamp < self.last_timestamp:
            self.tracks.clear()
        self.frame_size = (width,height)
        self.last_timestamp = timestamp
        for person in people:
            track_id = person['entity_local_id'] if isinstance(person,dict) else person.entity_local_id
            box = person['bbox_xyxy_px'] if isinstance(person,dict) else person.bbox_xyxy_px
            if person_context_crops([box],width,height,size):
                self.tracks[track_id] = (tuple(box),timestamp)
        self.tracks = {k:v for k,v in self.tracks.items() if timestamp-v[1] <= self.hold_seconds}
        regions = []
        for track_id,(box,seen) in sorted(self.tracks.items()):
            for region in person_context_crops([box],width,height,size):
                regions.append({'crop_xyxy':region,'proposal_age_sec':max(0.,timestamp-seen),'proposal_entity_id':track_id})
        return regions


def detect_phone_candidates(model, frame, people_boxes, *, device='cuda:0', full_image_size=1280, confidence=.25, crop_size=384, use_crops=True, regions_override=None):
    """Shared evaluator/service implementation. All confidence values stay raw."""
    kwargs = {'conf':confidence,'verbose':False}
    if str(device).startswith('cuda'):
        kwargs['device'] = device
    result = model.predict(frame,imgsz=full_image_size,**kwargs)[0]
    objects = [{'object_class':'MOBILE_PHONE','confidence':box.conf.item(),'bbox_xyxy_px':box.xyxy[0].tolist(),'inference_view':'full_frame'} for box in result.boxes]
    if use_crops:
        height,width = frame.shape[:2]
        region_info = regions_override if regions_override is not None else [{'crop_xyxy':r,'proposal_age_sec':0.} for r in person_context_crops(people_boxes,width,height,crop_size)]
        unique = {}
        for info in region_info:
            key = tuple(info['crop_xyxy'])
            if key not in unique or info['proposal_age_sec'] < unique[key]['proposal_age_sec']:
                unique[key] = info
        region_info = list(unique.values())
        regions = [r['crop_xyxy'] for r in region_info]
        if regions:
            crops = [frame[y:y2,x:x2] for x,y,x2,y2 in regions]
            results = model.predict(crops,imgsz=640,**kwargs)
            for info,crop_result in zip(region_info,results):
                region = info['crop_xyxy']
                x,y,_,_ = region
                for box in crop_result.boxes:
                    bx,by,bx2,by2 = box.xyxy[0].tolist()
                    objects.append({'object_class':'MOBILE_PHONE','confidence':box.conf.item(),'bbox_xyxy_px':[bx+x,by+y,bx2+x,by2+y],'inference_view':'held_person_context_crop' if info['proposal_age_sec'] > 0 else 'person_context_crop','crop_xyxy':region,'crop_proposal_age_sec':info['proposal_age_sec'],'crop_proposal_entity_id':info.get('proposal_entity_id')})
    retained = []
    for obj in sorted(objects,key=lambda p:p['confidence'],reverse=True):
        if all(_iou(obj['bbox_xyxy_px'],r['bbox_xyxy_px']) < .5 for r in retained):
            retained.append(obj)
    return retained
