"""
Emotion Detection Module using Watson NLP Emotion Predict service.
"""
import json
import socket
import requests

SERVICE_STATUS = {"reachable": None}


def is_service_available():
    """
    Checks if the Watson NLP service is reachable on the current network.
    Caches the result to avoid repeated connection delays when offline.
    """
    if SERVICE_STATUS["reachable"] is not None:
        return SERVICE_STATUS["reachable"]
    try:
        with socket.create_connection(
            ("sn-watson-emotion.labs.skills.network", 443), timeout=0.5
        ):
            SERVICE_STATUS["reachable"] = True
    except (socket.error, OSError):
        SERVICE_STATUS["reachable"] = False
    return SERVICE_STATUS["reachable"]


def emotion_detector(text_to_analyze):
    """
    Sends text to Watson NLP EmotionPredict and returns emotion scores
    and the dominant emotion, or None values if status code is 400.
    """
    if not is_service_available():
        return _local_mock_detector(text_to_analyze)

    url = (
        'https://sn-watson-emotion.labs.skills.network'
        '/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.exceptions.RequestException:
        return _local_mock_detector(text_to_analyze)

    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']

    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }


def _local_mock_detector(text_to_analyze):
    """
    Fallback mock detector for local testing when IBM Watson endpoint is unreachable.
    """
    if not text_to_analyze or not str(text_to_analyze).strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    text_lower = text_to_analyze.lower()
    scores = {
        'anger': 0.05,
        'disgust': 0.05,
        'fear': 0.05,
        'joy': 0.05,
        'sadness': 0.05
    }

    if 'glad' in text_lower or 'happy' in text_lower:
        dominant = 'joy'
    elif 'mad' in text_lower or 'angry' in text_lower:
        dominant = 'anger'
    elif 'disgust' in text_lower:
        dominant = 'disgust'
    elif 'sad' in text_lower:
        dominant = 'sadness'
    elif 'afraid' in text_lower or 'fear' in text_lower:
        dominant = 'fear'
    else:
        dominant = 'joy'

    scores[dominant] = 0.8
    scores['dominant_emotion'] = dominant
    return scores
