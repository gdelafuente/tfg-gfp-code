"""
Global variables used as configuration for the tool
"""

from pathlib import Path
from nltk.corpus import wordnet as wn


VENDORS = ["amazon", "azure", "clarifai", "google", "watson"]
"""Vendors to be compared"""

PLACES_PATH = Path("datasets/places")
"""Path of the Places dataset"""

INDOOR_PATH = Path("datasets/indoor")
"""Path of the Indoor dataset"""

SUN_PATH = Path("datasets/sun")
"""Path of the SUN dataset"""

RESULTS_PATH = Path("results")
"""Path to store vendors data in"""

IMAGES_PER_CATEGORY = 5
"""Number of images per category to label"""

LABELS_PER_IMAGE = 5
"""Number of labels per image to use"""

SIMILARITY_THRESHOLD = 0.80
"""Minimum similarity with the ground truth to consider a label as correct"""

DISCARDED_HYPERNYMS = [wn.synset('artifact.n.01'), wn.synset('object.n.01'), wn.synset('instrumentality.n.03'), wn.synset('entity.n.01'), 
                        wn.synset('whole.n.02'), wn.synset('depository.n.01'), wn.synset('area.n.05'), wn.synset('depository.n.01'), 
                        wn.synset('organization.n.01'), wn.synset('abstraction.n.06'), wn.synset('curve.n.01'), wn.synset('institution.n.01'), 
                        wn.synset('shape.n.02'), wn.synset('site.n.01'),wn.synset('floor.n.02'), wn.synset('state.n.02'), wn.synset('attribute.n.02'), 
                        wn.synset('communication.n.02'), wn.synset('matter.n.03'), wn.synset('geographic_point.n.01'), wn.synset('geographical_area.n.01'), 
                        wn.synset('tract.n.01'), wn.synset('physical_entity.n.01'), wn.synset('device.n.01'), wn.synset('structure.n.01'), 
                        wn.synset('establishment.n.04'), wn.synset('region.n.03'), wn.synset('psychological_feature.n.01'), wn.synset('event.n.01'), 
                        wn.synset('thing.n.12'), wn.synset('point.n.02'), wn.synset('location.n.01'), wn.synset('group.n.01'), wn.synset('surface.n.01'), 
                        wn.synset('social_group.n.01'), wn.synset('container.n.01'), wn.synset('activity.n.01'), wn.synset('measure.n.02'), 
                        wn.synset('compartment.n.02'), wn.synset('enclosure.n.01')]
"""Too ambiguous or out of scope parent categories"""

CREDENTIALS = {
    "azure": {
        "endpoint": "***",
        "key": "***"
    },
    "google": {
        "key": "***"
    },
    "clarifai": {
        "key": "***"
    },
    "watson": {
        "key": "***"
    },
    "amazon": {
        "id": "***",
        "key": "***"
    }
}
