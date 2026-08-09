import logging, time
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from src.ml.inference import load_model, predict_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('catdog-api')
app = FastAPI(title='Cats vs Dogs MLOps Inference API', version='1.0.0')
model = load_model()
metrics = {'request_count': 0, 'prediction_count': 0, 'total_latency_ms': 0.0}

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    start = time.time(); metrics['request_count'] += 1
    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
        result = predict_image(model, image)
        latency = (time.time() - start) * 1000
        metrics['prediction_count'] += 1; metrics['total_latency_ms'] += latency
        logger.info('prediction filename=%s label=%s latency_ms=%.2f', file.filename, result['label'], latency)
        return {'filename': file.filename, 'latency_ms': round(latency, 2), **result}
    except Exception as exc:
        logger.exception('prediction_failed filename=%s', file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get('/metrics')
def get_metrics():
    avg = metrics['total_latency_ms'] / metrics['prediction_count'] if metrics['prediction_count'] else 0.0
    return {**metrics, 'avg_latency_ms': round(avg, 2)}
