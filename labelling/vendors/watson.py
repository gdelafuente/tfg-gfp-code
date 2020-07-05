"""
IBM Watson Visual Recognition wrapper
"""
import logging
from typing import List, Tuple

from watson_developer_cloud import VisualRecognitionV3

from labelling.settings import CREDENTIALS
from labelling.vendors import LabelsGatherer


class WatsonLabelsGatherer(LabelsGatherer):

    def __init__(self):

        # Create client
        try:
            self.__client = VisualRecognitionV3('2018-03-19', iam_apikey=CREDENTIALS["watson"]["key"])
        except Exception as ex:
            logging.error(f"Error creating client: {ex}")


    def get_labels(self, image: str, n: int) -> Tuple[List[str], List[float]]:
        
        # Request server
        try:
            with open(image, 'rb') as image:
                response = self.__client.classify(images_file=image)
        except Exception as ex:
            logging.error(f"Error calling server: {ex}")
            return None

        # Process response
        labels = []
        scores = []
        try:
            for tag in response.result["images"][0]["classifiers"][0]["classes"]:
                labels.append(tag["class"])
                scores.append(tag["score"])
            return labels[:n], scores[:n]
        except Exception as ex:
            logging.error(f"Error processing server response: {ex}")
            return None, None
