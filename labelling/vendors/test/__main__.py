import os
import sys
import unittest

path = os.path.abspath(__file__ + "/../../../")
dir_path = os.path.dirname(path)
sys.path.insert(0, dir_path)

from labelling.vendors.amazon import AmazonLabelsGatherer
from labelling.vendors.azure import AzureLabelsGatherer
from labelling.vendors.clarifai import ClarifaiLabelsGatherer
from labelling.vendors.google import GoogleLabelsGatherer
from labelling.vendors.watson import WatsonLabelsGatherer


class TestLabeling(unittest.TestCase):
    """Simple unit tests to check that credentials are still valid
    """

    @classmethod
    def setUpClass(cls):
        """Set test image
        """
        cls.image_path = "labelling/vendors/test/resources/bikes.jpg"

    def test_azure(self):
        """Test that both a list of labels and a list of scores are returned
        """

        labels, scores = AzureLabelsGatherer().get_labels(self.image_path,1)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(labels[0], str)
        self.assertIsInstance(scores[0], float)

    def test_amazon(self):
        """Test that both a list of labels and a list of scores are returned
        """

        labels, scores = AmazonLabelsGatherer().get_labels(self.image_path, 1)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(labels[0], str)
        self.assertIsInstance(scores[0], float)

    def test_clarifai(self):
        """Test that both a list of labels and a list of scores are returned
        """

        labels, scores = ClarifaiLabelsGatherer().get_labels(self.image_path, 1)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(labels[0], str)
        self.assertIsInstance(scores[0], float)

    def test_watson(self):
        """Test that both a list of labels and a list of scores are returned
        """

        labels, scores = WatsonLabelsGatherer().get_labels(self.image_path, 1)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(labels[0], str)
        self.assertIsInstance(scores[0], float)

    def test_google(self):
        """Test that both a list of labels and a list of scores are returned
        """

        labels, scores = GoogleLabelsGatherer().get_labels(self.image_path, 1)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(labels[0], str)
        self.assertIsInstance(scores[0], float)


if __name__ == '__main__':
    unittest.main()
