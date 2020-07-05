"""
Amazon Web Services Rekognition wrapper
"""
import logging
from typing import List, Tuple

import boto3

from labelling.settings import CREDENTIALS
from labelling.vendors import LabelsGatherer


class AmazonLabelsGatherer(LabelsGatherer):

    def __init__(self):

        # Create client
        try:
            self.__client = boto3.client(
                'rekognition',
                region_name='eu-west-2',
                aws_access_key_id=CREDENTIALS["amazon"]["id"],
                aws_secret_access_key=CREDENTIALS["amazon"]["key"]
            )
        except Exception as ex:
            logging.error(f"Error creating client: {ex}")


    def get_labels(self, image: str, n: int) -> Tuple[List[str], List[float]]:

        # Load image as bytes
        with open(image, "rb") as image:
            image_data = image.read()

        # Request server
        try:
            response = self.__client.detect_labels(Image={'Bytes': image_data})
        except Exception as ex:
            logging.error(f"Error calling server: {ex}")
            return None

        # Process response
        labels = []
        scores = []
        try:
            for tag in response["Labels"]:
                labels.append(tag["Name"])
                scores.append(tag["Confidence"])
            return (labels[:n], scores[:n])
        except Exception as ex:
            logging.error(f"Error processing server response: {ex}")
            return None, None
