import os
import io
import json
import asyncio
import time
from typing import List, Optional, Set

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse

try:
	import tflite_runtime.interpreter as tflite  # type: ignore
	TFLITE_AVAILABLE = True
except Exception:
	try:
		import tensorflow as tf  # type: ignore
		tflite = tf.lite  # fallback to TF's bundled TFLite
		TFLITE_AVAILABLE = True
	except Exception:
		TFLITE_AVAILABLE = False


app = FastAPI(title="Plant Health Inference API", version="0.1.0")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "mobilenet_v2.tflite"))
LABELS_PATH = os.path.abspath(os.path.join(MODEL_DIR, "labels.txt"))
API_KEY = os.getenv("API_KEY", "").strip()

_interpreter = None
_input_details = None
_output_details = None
_labels: Optional[List[str]] = None
_input_size = (224, 224)

# Live state and SSE subscribers
_latest_status: dict = {"timestamp": None, "label": None, "probability": None}
_subscribers: Set[asyncio.Queue] = set()


def load_labels(path: str) -> Optional[List[str]]:
	if not os.path.exists(path):
		return None
	with open(path, "r", encoding="utf-8") as f:
		labels = [line.strip() for line in f if line.strip()]
	return labels or None


def initialize_model() -> None:
	global _interpreter, _input_details, _output_details, _labels, _input_size

	if not TFLITE_AVAILABLE:
		raise RuntimeError("TFLite not available. Install tflite-runtime or tensorflow.")

	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(
			f"Model not found at '{MODEL_PATH}'. Place your TFLite model there."
		)

	# Ensure model dir exists
	os.makedirs(MODEL_DIR, exist_ok=True)

	# Load labels if available
	_labels = load_labels(LABELS_PATH)

	# Initialize interpreter
	try:
		if hasattr(tflite, "Interpreter"):
			interpreter = tflite.Interpreter(model_path=MODEL_PATH)
		else:
			interpreter = tflite.interpreter.Interpreter(model_path=MODEL_PATH)  # type: ignore
	except Exception as e:
		raise RuntimeError(f"Failed to load TFLite model: {e}")

	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Assume single input
	input_shape = input_details[0]["shape"]
	# Expected shape: [1, height, width, channels]
	if len(input_shape) == 4:
		_input_size = (int(input_shape[2]), int(input_shape[1]))  # (w, h)
	else:
		_input_size = (224, 224)

	_interpreter = interpreter
	_input_details = input_details
	_output_details = output_details


@app.on_event("startup")
def _on_startup() -> None:
	# Lazy-init model. If missing, health endpoint will reflect not-ready.
	try:
		initialize_model()
	except Exception:
		pass


@app.get("/health")
async def health() -> JSONResponse:
	model_loaded = _interpreter is not None
	details = {
		"model_loaded": model_loaded,
		"model_path": MODEL_PATH,
		"labels_path": LABELS_PATH,
		"labels_count": len(_labels) if _labels else 0,
		"input_size": _input_size,
		"latest": _latest_status,
	}
	status = 200 if model_loaded else 503
	return JSONResponse(content=details, status_code=status)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	image = image.resize(_input_size)
	image_np = np.asarray(image).astype(np.float32)
	# MobileNetV2 expects inputs in [-1, 1]
	image_np = (image_np / 127.5) - 1.0
	image_np = np.expand_dims(image_np, axis=0)  # [1, H, W, C]
	return image_np


def infer(image_tensor: np.ndarray) -> np.ndarray:
	assert _interpreter is not None and _input_details and _output_details
	_interpreter.set_tensor(_input_details[0]["index"], image_tensor)
	_interpreter.invoke()
	output = _interpreter.get_tensor(_output_details[0]["index"])  # [1, num_classes]
	return output[0]


def postprocess(scores: np.ndarray) -> dict:
	# Softmax if not applied
	if scores.ndim == 1:
		# Numerical stability
		scores_exp = np.exp(scores - np.max(scores))
		probs = scores_exp / np.sum(scores_exp)
	else:
		probs = scores

	top_idx = int(np.argmax(probs))
	top_prob = float(probs[top_idx])

	if _labels and 0 <= top_idx < len(_labels):
		top_label = _labels[top_idx]
	else:
		# Fallback to generic names
		top_label = f"class_{top_idx}"

	return {
		"label": top_label,
		"probability": round(top_prob, 4),
		"scores": [float(x) for x in probs.tolist()],
		"labels": _labels or [],
	}


def _publish_update(event: str, data: dict) -> None:
	"""Publish an SSE event to all subscribers and update latest status."""
	_latest_status.update({
		"timestamp": int(time.time()),
		"label": data.get("label"),
		"probability": data.get("probability"),
	})
	message = json.dumps({"event": event, "data": data, "ts": _latest_status["timestamp"]})
	for q in list(_subscribers):
		try:
			q.put_nowait(message)
		except Exception:
			pass


@app.get("/events")
async def sse_events() -> StreamingResponse:
	async def event_generator(queue: asyncio.Queue):
		try:
			# Send initial snapshot
			init = json.dumps({"event": "snapshot", "data": _latest_status, "ts": int(time.time())})
			yield f"data: {init}\n\n"
			while True:
				payload = await queue.get()
				yield f"data: {payload}\n\n"
		except asyncio.CancelledError:
			return

	queue: asyncio.Queue = asyncio.Queue()
	_subscribers.add(queue)
	response = StreamingResponse(event_generator(queue), media_type="text/event-stream")

	@response.call_on_close
	def _cleanup() -> None:
		try:
			_subscribers.remove(queue)
		except KeyError:
			pass

	return response


@app.get("/dashboard")
async def dashboard() -> HTMLResponse:
	html = """
	<!doctype html>
	<html lang=\"en\">
	<head>
		<meta charset=\"utf-8\">
		<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
		<title>AI Sprayer Dashboard</title>
		<style>
			body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 16px; background: #0b1220; color: #e9edf1; }
			.card { background: #121a2a; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
			.row { display: flex; gap: 12px; }
			.row > .card { flex: 1; }
			.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; }
			.badge.ok { background: #123f28; color: #9ef7c0; }
			.badge.warn { background: #3f2e12; color: #f7d29e; }
			.badge.danger { background: #3f1212; color: #f7a3a3; }
			.small { color: #98a2b3; font-size: 12px; }
			.value { font-size: 32px; font-weight: 700; }
			.label { font-size: 14px; color: #b6c2d2; }
			footer { font-size: 12px; color: #7a8799; margin-top: 8px; }
			button { background: #2b6cb0; color: white; border: 0; padding: 10px 14px; border-radius: 8px; font-weight: 600; }
			@media (max-width: 600px){ .row { flex-direction: column; } }
		</style>
	</head>
	<body>
		<h2>Intelligent Pesticide Sprinkling System</h2>
		<div class=\"row\">
			<div class=\"card\">
				<div class=\"label\">Current Status</div>
				<div id=\"status\" class=\"value\">—</div>
				<div id=\"prob\" class=\"label\">Confidence: —</div>
				<div id=\"badge\" class=\"badge\">No Data</div>
				<div class=\"small\">Updated: <span id=\"ts\">—</span></div>
			</div>
		</div>
		<div class=\"card\" style=\"margin-top:12px\">
			<button id=\"refresh\">Refresh Health</button>
			<pre id=\"health\" class=\"small\" style=\"white-space: pre-wrap\"></pre>
		</div>
		<footer>Live updates via SSE. Open this page on your phone on the same Wi‑Fi.</footer>
		<script>
		function fmtTs(ts){ if(!ts) return '—'; const d=new Date(ts*1000); return d.toLocaleTimeString(); }
		function setBadge(label){
			const el=document.getElementById('badge');
			el.className='badge';
			if(label==='Healthy') el.classList.add('ok');
			else if(label==='Mild') el.classList.add('warn');
			else if(label==='Severe') el.classList.add('danger');
			el.textContent=label||'No Data';
		}
		function apply(data){
			document.getElementById('status').textContent = data.label ?? '—';
			document.getElementById('prob').textContent = 'Confidence: ' + (data.probability?.toFixed ? data.probability.toFixed(2) : data.probability ?? '—');
			setBadge(data.label);
			document.getElementById('ts').textContent = fmtTs(data.timestamp || data.ts);
		}
		const es = new EventSource('/events');
		es.onmessage = (ev)=>{
			try{
				const msg = JSON.parse(ev.data);
				if(msg.event==='prediction') apply({ ...msg.data, ts: msg.ts });
				if(msg.event==='snapshot') apply(msg.data);
			}catch(e){ console.error(e); }
		};
		document.getElementById('refresh').onclick = async ()=>{
			try{ const r = await fetch('/health'); const j = await r.json(); document.getElementById('health').textContent = JSON.stringify(j,null,2); }
			catch(e){ document.getElementById('health').textContent = 'Health fetch failed'; }
		};
		</script>
	</body>
	</html>
	"""
	return HTMLResponse(content=html)


@app.post("/predict")
async def predict(
	file: UploadFile = File(...),
	api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> JSONResponse:
	# API key check (enabled only if API_KEY is configured)
	if API_KEY:
		if api_key is None or api_key.strip() != API_KEY:
			raise HTTPException(status_code=401, detail="Unauthorized")
	if _interpreter is None:
		# Try to initialize now (allows starting server before placing model)
		try:
			initialize_model()
		except Exception as e:
			raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

	if file.content_type is None or not file.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Please upload an image file.")

	image_bytes = await file.read()
	try:
		image_tensor = preprocess_image(image_bytes)
		scores = infer(image_tensor)
		result = postprocess(scores)
		_publish_update("prediction", result)
		return JSONResponse(content=result)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Inference error: {e}")
