import os
import re
import base64
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from face_utils import get_embedding, cosine_similarity

__version__ = "1.0.0" 


# ─── Configuration ───────────────────────────────────────────────────────
PORT          = 8080
STATIC_FOLDER = 'static'
ALLOWED_EXT   = {'png', 'jpg', 'jpeg'}
# ──────────────────────────────────────────────────────────────────────────

def secure_filename(fn: str) -> str:
    return os.path.basename(fn).replace(' ', '_')

def load_static(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()

def render_html(template: str, context: dict) -> str:
    for key, val in context.items():
        template = template.replace(f'{{{{{key}}}}}', str(val))
    return template

def label_and_class(score: float):
    if score > 0.55:
        return 'Extremely possible', 'likelihood-extremely'
    if score > 0.50:
        return 'Highly likely',    'likelihood-highly'
    if score > 0.45:
        return 'Likely',           'likelihood-likely'
    return 'Unlikely', 'likelihood-unlikely'

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve CSS or HTML
        if self.path.startswith('/static/'):
            file_path = self.path.lstrip('/')
            if os.path.isfile(file_path):
                self.send_response(200)
                mime, _ = mimetypes.guess_type(file_path)
                self.send_header('Content-Type', mime or 'application/octet-stream')
                self.end_headers()
                self.wfile.write(load_static(file_path))
                return
            return self.send_error(404)

        # Default: serve index.html
        html = load_static(os.path.join(STATIC_FOLDER, 'index.html')).decode('utf-8')
        output = render_html(html, {'error_block':'', 'result_block':'', 'version': __version__})
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(output.encode('utf-8'))

    def do_POST(self):
        ctype = self.headers.get('Content-Type','')
        if 'multipart/form-data' not in ctype:
            return self._error('Invalid content type.')

        boundary = ctype.split('boundary=')[-1].encode()
        length   = int(self.headers.get('Content-Length',0))
        body     = self.rfile.read(length)

        # Parse uploaded files
        parts = body.split(b'--'+boundary)
        images = {}
        for part in parts:
            if b'Content-Disposition' not in part:
                continue
            hdr, data = part.split(b'\r\n\r\n',1)
            data = data.rstrip(b'\r\n--')
            disp = hdr.decode(errors='ignore')
            nm = re.search(r'name="([^"]+)"', disp)
            fn = re.search(r'filename="([^"]+)"', disp)
            if not (nm and fn):
                continue

            field = nm.group(1)
            raw  = fn.group(1)
            ext  = raw.rsplit('.',1)[-1].lower()
            if ext not in ALLOWED_EXT:
                return self._error('Allowed file types: png, jpg, jpeg')

            b64 = base64.b64encode(data).decode('ascii')
            mime = f'image/{"jpeg" if ext in ("jpg","jpeg") else ext}'
            images[field] = f'data:{mime};base64,{b64}'

        if 'img1' not in images or 'img2' not in images:
            return self._error('Please upload both images.')

        # Write temps for embedding
        tmp1 = os.path.join(STATIC_FOLDER, secure_filename('tmp1.png'))
        tmp2 = os.path.join(STATIC_FOLDER, secure_filename('tmp2.png'))
        with open(tmp1,'wb') as f: f.write(base64.b64decode(images['img1'].split(',')[1]))
        with open(tmp2,'wb') as f: f.write(base64.b64decode(images['img2'].split(',')[1]))

        try:
            e1 = get_embedding(tmp1)
            e2 = get_embedding(tmp2)
            score = cosine_similarity(e1, e2)
        except Exception as e:
            return self._error(str(e))
        finally:
            os.remove(tmp1)
            os.remove(tmp2)

        likelihood, css_class = label_and_class(score)

        result_html = f"""
<div class="result">
  <div class="images">
    <img src="{images['img1']}" alt="Image 1"/>
    <img src="{images['img2']}" alt="Image 2"/>
  </div>
  <h3 class="score">Cosine Similarity: {score:.4f}</h3>
  <h3 class="{css_class}">Likelihood: {likelihood}</h3>
</div>"""

        html = load_static(os.path.join(STATIC_FOLDER, 'index.html')).decode('utf-8')
        output = render_html(html, {'error_block':'', 'result_block':result_html, 'version': __version__})
        self.send_response(200)
        self.send_header('Content-Type','text/html')
        self.end_headers()
        self.wfile.write(output.encode('utf-8'))

    def _error(self, msg: str):
        html = load_static(os.path.join(STATIC_FOLDER, 'index.html')).decode('utf-8')
        err = f'<div class="error">{msg}</div>'
        output = render_html(html, {'error_block':err, 'result_block':'', 'version': __version__})
        self.send_response(400)
        self.send_header('Content-Type','text/html')
        self.end_headers()
        self.wfile.write(output.encode('utf-8'))

def run():
    print(f"Serving at http://localhost:{PORT}")
    HTTPServer(('', PORT), Handler).serve_forever()

if __name__=='__main__':
    run()
