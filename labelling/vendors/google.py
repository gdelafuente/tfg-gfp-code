"""
Google Cloud Computer Vision wrapper
"""
import base64
import logging
from typing import List, Tuple

import requests

from labelling.settings import CREDENTIALS
from labelling.vendors import LabelsGatherer


class GoogleLabelsGatherer(LabelsGatherer):

    def __init__(self):
        pass

    def get_labels(self, image: str, n: int) -> Tuple[List[str], List[float]]:

        # Load image as base64
        with open(image, "rb") as image:
            image_data = base64.b64encode(image.read()).decode()

        # Request server
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_data
                    },
                    "features": [
                        {
                            "type": "LABEL_DETECTION",
                            "maxResults": 10
                        },
                    ]
                }
            ]
        }
        try:
            response = requests.post("https://vision.googleapis.com/v1/images:annotate?key=" + CREDENTIALS["google"]["key"], json=payload)
            response.raise_for_status()
        except Exception as ex:
            logging.error(f"Error calling server: {ex}")
            return None

        # Process response
        labels = []
        scores = []
        try:
            for tag in response.json()["responses"][0]["labelAnnotations"]:
                labels.append(tag["description"])
                scores.append(tag["score"])
            return labels[:n], scores[:n]
        except Exception as ex:
            logging.error(f"Error processing server response: {ex}")
            return None, None
