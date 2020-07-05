"""
Azure Computer Vision wrapper
"""
import logging
from typing import List, Tuple

import requests

from labelling.settings import CREDENTIALS
from labelling.vendors import LabelsGatherer


class AzureLabelsGatherer(LabelsGatherer):

    def __init__(self):
        pass

    def get_labels(self, image: str, n: int) -> Tuple[List[str], List[float]]:

        # Load image as bytes
        with open(image, "rb") as image:
            image_data = image.read()

        # Request server
        try:
            response = requests.post(
                CREDENTIALS["azure"]["endpoint"] + "/vision/v2.0/tag",
                headers={
                    'Ocp-Apim-Subscription-Key': CREDENTIALS["azure"]["key"],
                    'Content-Type': 'application/octet-stream'
                },
                data=image_data
            )
            response.raise_for_status()
        except Exception as ex:
            logging.error(f"Error calling server: {ex}")
            return None

        # Process response
        labels = []
        scores = []
        try:
            for tag in response.json()["tags"]:
                labels.append(tag["name"])
                scores.append(tag["confidence"])
            return labels[:n], scores[:n]
        except Exception as ex:
            logging.error(f"Error processing server response: {ex}")
            return None, None
