"""Local-only manual proposal review. No annotation edits or training endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
from uuid import uuid4

STATUSES = {'UNKNOWN','CONFIRMED_PHONE','NOT_PHONE','AMBIGUOUS'}


def validate_decision(payload, proposals):
    if payload.get('manual_ack') is not True:
        raise ValueError('Explicit human review acknowledgement required')
    if payload.get('status') not in STATUSES:
        raise ValueError('Unsupported status; UNKNOWN is never a negative')
    reviewer = str(payload.get('reviewer','')).strip()
    if not reviewer or len(reviewer)>200:
        raise ValueError('Human reviewer ID required (1–200 characters)')
    ids = payload.get('proposal_ids')
    if not isinstance(ids,list) or not ids or any(not isinstance(p,str) for p in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Provide a non-empty list of unique proposal IDs')
    if any(p not in proposals for p in ids):
        raise ValueError('Unknown proposal ID')
    if len(ids)>1 and payload.get('all_proposals_reviewed') is not True:
        raise ValueError('Every selected proposal must be manually reviewed')
    if len({proposals[p]['segment_id'] for p in ids}) != 1:
        raise ValueError('One segment per manual decision transaction')
    return {'proposal_ids':ids,'status':payload['status'],'reviewer':reviewer,
        'notes':str(payload.get('notes',''))[:10000], 'manual_ack':True,
        'all_proposals_reviewed':len(ids)==1 or payload.get('all_proposals_reviewed') is True}


class ReviewStore:
    def __init__(self, package):
        self.package = Path(package).resolve()
        data_path = self.package/'review_data.json'
        self.fingerprint = sha256(data_path.read_bytes()).hexdigest()
        ready = json.loads((self.package/'READY_FOR_HUMAN_REVIEW.json').read_text())
        if ready['review_data_sha256'] != self.fingerprint:
            raise ValueError('Review package has changed; refusing to apply decisions')
        data = json.loads(data_path.read_text(encoding='utf-8'))
        self.proposals = {p['proposal_id']:p for p in data['proposals']}
        if len(self.proposals)!=len(data['proposals']):
            raise ValueError('Duplicate proposal IDs')
        self.path = self.package/'manual_decisions.jsonl'
        self.lock = threading.Lock()
        self.history = []
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                validate_decision(row,self.proposals)
                if row['review_data_sha256']!=self.fingerprint:
                    raise ValueError('Decision provenance mismatch')
                self.history.append(row)

    def state(self):
        with self.lock:
            statuses = {p:'UNKNOWN' for p in self.proposals}
            for row in self.history:
                for p in row['proposal_ids']:
                    statuses[p] = row['status']
            return {'statuses':statuses,'decision_transactions':len(self.history),
                'training_authorized':False,'review_data_sha256':self.fingerprint}

    def decide(self, payload):
        decision = validate_decision(payload,self.proposals)
        decision.update(transaction_id=str(uuid4()),reviewed_at=datetime.now(timezone.utc).isoformat(),
            review_data_sha256=self.fingerprint,training_authorized=False)
        with self.lock:
            with self.path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(decision)+'\n')
                stream.flush()
                import os
                os.fsync(stream.fileno())
            self.history.append(decision)
        return decision


def handler_for(store, token):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,directory=str(store.package),**kwargs)

        def response(self,payload,status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path=='/api/config':
                self.response({'token':token,'statuses':sorted(STATUSES)})
            elif path=='/api/state':
                self.response(store.state())
            elif path=='/favicon.ico':
                self.send_response(204)
                self.end_headers()
            else:
                super().do_GET()

        def list_directory(self,path):
            self.send_error(403,'Directory listing disabled')
            return None

        def send_head(self):
            # Serve only the generated package; never adjacent repo/source files.
            translated = Path(self.translate_path(self.path)).resolve()
            try:
                translated.relative_to(store.package)
            except ValueError:
                self.send_error(403)
                return None
            if translated.name in ('manual_decisions.jsonl',):
                self.send_error(403,'Use review state endpoint')
                return None
            if urlsplit(self.path).path=='/preview.mp4' and translated.is_file():
                size = translated.stat().st_size
                start,end = 0,size-1
                header = self.headers.get('Range')
                if header:
                    try:
                        if not header.startswith('bytes=') or ',' in header:
                            raise ValueError()
                        left,right = header[6:].split('-')
                        if left:
                            start=int(left)
                            end=min(size-1,int(right)) if right else size-1
                        else:
                            count=int(right)
                            if count<=0: raise ValueError()
                            start=max(0,size-count)
                        if start<0 or start>=size or end<start: raise ValueError()
                    except (ValueError,TypeError):
                        self.send_response(416)
                        self.send_header('Content-Range',f'bytes */{size}')
                        self.end_headers()
                        return None
                self.send_response(206 if header else 200)
                self.send_header('Content-Type','video/mp4')
                self.send_header('Accept-Ranges','bytes')
                self.send_header('Content-Length',str(end-start+1))
                if header: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                stream = translated.open('rb')
                stream.seek(start)
                self.remaining = end-start+1
                return stream
            self.remaining = None
            return super().send_head()

        def copyfile(self,source,output):
            try:
                if getattr(self,'remaining',None) is None:
                    return super().copyfile(source,output)
                remaining = self.remaining
                while remaining:
                    block = source.read(min(65536,remaining))
                    if not block: break
                    output.write(block)
                    remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):
                # Browsers cancel the old media range request when seeking.
                return

        def do_POST(self):
            if urlsplit(self.path).path!='/api/decision':
                self.send_error(404)
                return
            expected = {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if self.headers.get('Origin') not in expected or self.headers.get('X-Review-Token')!=token:
                self.response({'error':'Local origin and review token required'},403)
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1000000:
                    raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError('Object payload required')
                self.response({'decision':store.decide(payload)})
            except (ValueError,KeyError,TypeError) as error:
                self.response({'error':str(error)},400)
            except OSError:
                self.response({'error':'Decision could not be persisted; no success recorded'},500)
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    store=ReviewStore(args.package)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(store,secrets.token_urlsafe(32)))
    print(f'Human review: http://127.0.0.1:{server.server_port}; Ctrl+C to stop. No training or annotation edits.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
