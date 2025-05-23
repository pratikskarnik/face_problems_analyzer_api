# Face Problems Analyzer API

This API allows users to upload a face image and get predictions for common skin problems using a deep learning model.

## Endpoints

### `POST /predict`
**Input**: Image file (JPG/PNG)  
**Output**: JSON with predicted skin condition and confidence score

## Example Request

```bash
curl -X POST http://localhost:8000/predict -F "file=@your_image.jpg"
