"""Read-only headless browser smoke check; never submits manual review decisions."""
import argparse
from hashlib import sha256
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parent))
from serve_video03_human_review import ReviewStore,handler_for


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_segments=False
        self.in_counts=False
        self.in_error=False
        self.segments=0
        self.counts=''
        self.error=''
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='select' and attrs.get('id')=='segments': self.in_segments=True
        if tag=='option' and self.in_segments: self.segments+=1
        if attrs.get('id')=='counts': self.in_counts=True
        if attrs.get('id')=='error': self.in_error=True
    def handle_endtag(self,tag):
        if tag=='select': self.in_segments=False
        if tag=='p': self.in_counts=False; self.in_error=False
    def handle_data(self,data):
        if self.in_counts: self.counts+=data
        if self.in_error: self.error+=data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chrome',default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    store=ReviewStore(args.package)
    before=store.state()
    ledger_hash=sha256(store.path.read_bytes()).hexdigest() if store.path.exists() else None
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(store,secrets.token_urlsafe(32)))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        result=subprocess.run([args.chrome,'--headless=new','--disable-gpu','--no-first-run',
            '--no-default-browser-check','--disable-background-networking',
            f'--user-data-dir={output / "chrome_profile"}','--window-size=1440,1200',
            f'--screenshot={output / "review_page.png"}','--dump-dom','--virtual-time-budget=8000',
            f'http://127.0.0.1:{server.server_port}'],capture_output=True,text=True,
            encoding='utf-8',errors='replace',timeout=45,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (output/'browser_stderr.txt').write_text(result.stderr,encoding='utf-8')
        if result.returncode: raise RuntimeError(f'Chrome returned {result.returncode}: {result.stderr[:500]}')
        (output/'rendered_page.html').write_text(result.stdout,encoding='utf-8')
        page=PageParser()
        page.feed(result.stdout)
        if page.error or page.segments!=22 or '849 proposals' not in page.counts:
            raise RuntimeError(f'Review page did not initialize: {page.segments} segments, {page.counts}, {page.error}')
        if store.state()!=before or (sha256(store.path.read_bytes()).hexdigest() if store.path.exists() else None)!=ledger_hash:
            raise RuntimeError('Read-only UI check changed review state')
        verified={'browser_page_loaded':True,'segments_rendered':page.segments,'counts_text':page.counts,
            'review_decisions_submitted':0,'review_state_unchanged':True,
            'note':'Rendering smoke check, not a comprehensive interactive-browser test'}
        (output/'verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
        print(json.dumps(verified,indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__=='__main__':
    main()
