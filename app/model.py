from fastai.vision.all import load_learner, PILImage

learn = load_learner("model/export.pkl")

async def predict_face_problem(file):
    image = PILImage.create(await file.read())
    pred, pred_idx, probs = learn.predict(image)
    return {
        "prediction": pred,
        "confidence": round(float(probs[pred_idx]), 4),
        "all_probs": {learn.dls.vocab[i]: round(float(p), 4) for i, p in enumerate(probs)}
    }
