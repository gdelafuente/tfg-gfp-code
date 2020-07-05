"""
Clarifai Predict wrapper
"""
import logging
from typing import List, Tuple

import yaml

from clarifai.rest import ClarifaiApp
from labelling.settings import CREDENTIALS
from labelling.vendors import LabelsGatherer


class ClarifaiLabelsGatherer(LabelsGatherer):

    def __init__(self):

        # Create client
        try:
            client = ClarifaiApp(api_key=CREDENTIALS["clarifai"]["key"])
            self.__model = client.public_models.general_model
        except Exception as ex:
            logging.error(f"Error creating client: {ex}")


    def get_labels(self, image: str, n: int) -> Tuple[List[str], List[float]]:

        # Load image as bytes
        with open(image, "rb") as image:
            image_data = image.read()

        # Request server
        try:
            response = self.__model.predict_by_bytes(image_data)
        except Exception as ex:
            logging.error(f"Error calling server: {ex}")
            return None

        # Process response
        labels = []
        scores = []
        try:
            for tag in response["outputs"][0]["data"]["concepts"]:
                labels.append(tag["name"])
                scores.append(tag["value"])
            return labels[:n], scores[:n]
        except Exception as ex:
            logging.error(f"Error processing server response: {ex}")
            return None, None
