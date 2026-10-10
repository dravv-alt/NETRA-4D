"""Real phone-only inference, spatial association and explicit crop provenance."""
from pathlib import Path

from .phone_crop import detect_phone_candidates, PersonCropMemory
from .object_provider import ObjectDisambiguationProvider
from ..schemas import ObjectClassEnum, ObjectObservation


class PhoneReviewProvider:
    def __init__(self, model_path, *, device, confidence=.25, roi_hold_seconds=1.):
        self.model_path = str(Path(model_path))
        self.device = device
        self.confidence = confidence
        self.model = None
        self.crop_memory = PersonCropMemory(roi_hold_seconds)

    def start(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        if set(self.model.names.values()) != {'phone'}:
            raise ValueError(f'Phone-only provider class-map mismatch: {self.model.names}')

    def close(self):
        self.model = None
        self.crop_memory.tracks.clear()

    def process_frame(self, frame_index, timestamp_sec, frame, people, *, camera_id, video_id):
        if self.model is None:
            raise RuntimeError('Phone provider has not started')
        height,width = frame.shape[:2]
        regions = self.crop_memory.regions(people,width,height,timestamp_sec)
        raw = detect_phone_candidates(self.model,frame,[p.bbox_xyxy_px for p in people],device=self.device,confidence=self.confidence,regions_override=regions)
        objects = []
        for row in raw:
            x1,y1,x2,y2 = row['bbox_xyxy_px']
            norm = (x1/width,y1/height,x2/width,y2/height)
            entity,seat,score,method = ObjectDisambiguationProvider._associate(norm,people)
            objects.append(ObjectObservation(camera_id=camera_id,video_id=video_id,frame_index=frame_index,timestamp_sec=timestamp_sec,object_class=ObjectClassEnum.MOBILE_PHONE,confidence=row['confidence'],bbox_xyxy_px=(x1,y1,x2,y2),bbox_xyxy_norm=norm,associated_entity_id=entity,associated_seat_id=seat,association_confidence=score,metadata={'raw_model_class':'phone','model_weight':self.model_path,'device':self.device,'inference_view':row['inference_view'],'crop_xyxy':row.get('crop_xyxy'),'crop_proposal_age_sec':row.get('crop_proposal_age_sec'),'crop_proposal_entity_id':row.get('crop_proposal_entity_id'),'association_method':method,'confidence_semantics':'raw detector confidence; not a behavioural probability'}))
        return objects
