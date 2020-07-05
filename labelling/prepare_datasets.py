"""
Datasets categorization using WordNet
"""
import os
import sys
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

path = os.path.abspath(__file__ + "/../")
dir_path = os.path.dirname(path)
sys.path.insert(0, dir_path)

from labelling.settings import DISCARDED_HYPERNYMS, RESULTS_PATH, VENDORS


def correct_indoor_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Correct Indoor ground truth

    Add some missing spaces, remove inside and fix some spelling errors
    """

    df["category"] = df["groundTruth"]
    df["groundTruth"] = df["groundTruth"].replace({
        "airport_inside": "airport",
        "artstudio": "art_studio",
        "bookstore": "book_store",
        "church_inside": "church",
        "clothingstore": "clothing_store",
        "computerroom": "computer_room",
        "dentaloffice": "dental_office",
        "gameroom": "game_room",
        "grocerystore": "grocery_store",
        "hairsalon": "hair_salon",
        "hospitalroom": "hospital_room",
        "inside_bus": "bus",
        "jewelleryshop": "jewellery_shop",
        "livingroom": "living_room",
        "movietheater": "movie_theater",
        "studiomusic": "music_studio",
        "poolinside": "pool",
        "prisoncell": "cell",
        "shoeshop": "shoe_shop",
        "stairscase": "staircase",
        "toystore": "toy_store",
        "trainstation": "train_station",
        "videostore": "video_store",
        "waitingroom": "waiting_room",
        "winecellar": "wine_cellar"
    })
    return df

def correct_sun_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Correct SUN ground truth

    Remove _indoor/_outdoor suffix
    """

    df["category"] = df["groundTruth"]
    df["indoor"] = df["groundTruth"].str.contains("_indoor")
    df["groundTruth"] = df["groundTruth"].apply(lambda x: x.replace("_indoor", "").replace("_outdoor", ""))
    return df


def calculate_categorization(df: pd.DataFrame) -> dict:
    """
    Returns a dict: {parent category: [child category]}.
    """

    # Calculate the first sustantive synset of every category
    synsets = {}
    bad_categories = []
    for category in df["groundTruth"].unique():
        try:
            synsets[category] = wn.synset(f"{category}.n.01")
        except:
            bad_categories.append(category)
    print(f"{len(bad_categories)} categories among {len(df['groundTruth'].unique())} have no synsets associated.")
    print(f"{bad_categories} were discarded.")

    # Generate a dictionary with the lowest hypernym in common with the rest of categories
    categories_hypernyms = {category : [] for category in synsets.keys()}
    for (category, synset) in synsets.items():
        for other_synset in synsets.values():
            categories_hypernyms[category].append(synset.lowest_common_hypernyms(other_synset)[0])   # Only first common hypernym

    # Select hypernyms with more than three hyponyms
    parent_categories = []
    hypernyms = [hypernym for sublist in categories_hypernyms.values() for hypernym in sublist]
    for (hypernym, count) in Counter(hypernyms).items():
        if count >= 3**2 and hypernym not in DISCARDED_HYPERNYMS:
            parent_categories.append(hypernym)

    # Get sons of these parent categories
    new_categories = {}
    for parent_category in parent_categories:
        new_categories[str(parent_category)] = []
        for (category, parents) in categories_hypernyms.items():
            try:
                if parent_category in parents:
                    new_categories[str(parent_category)].append(str(category))
            except:
                pass
    
    return new_categories


def apply_categorization(df: pd.DataFrame, categorization: dict) -> pd.DataFrame:
    """
    Replace categories ground truth by their parent category ground truth
    """

    pd.options.mode.chained_assignment = None   # Avoid pandas warning 
    new_categories = [child_category for child_categories in categorization.values() for child_category in child_categories]
    df = df.query(f"groundTruth in {new_categories}")
    for parent_category, child_categories in categorization.items():
        for child_category in child_categories:
            df.loc[df["groundTruth"] == child_category, 'groundTruth'] = parent_category
    return df


if __name__ == "__main__":

    print("---------INDOOR---------")
    indoor = pd.read_csv(RESULTS_PATH / "indoor.csv")
    indoor = correct_indoor_ground_truth(indoor)
    indoor.to_csv(RESULTS_PATH / "indoor.csv", index=False)
    categorization = calculate_categorization(indoor)
    for parent_category, child_categories in categorization.items():
        print(f"{parent_category}: {len(child_categories)} childs --> {child_categories}")

    print("---------PLACES---------")
    places = pd.read_csv(RESULTS_PATH / "places.csv")
    places["category"] = places["groundTruth"]
    places.to_csv(RESULTS_PATH / "places.csv", index=False)
    categorization = calculate_categorization(places)
    for parent_category, child_categories in categorization.items():
        print(f"{parent_category}: {len(child_categories)} childs --> {child_categories}")

    print("---------SUN---------")
    sun = pd.read_csv(RESULTS_PATH / "sun.csv")
    sun = correct_sun_ground_truth(sun)
    sun.to_csv(RESULTS_PATH / "sun.csv", index=False)
    categorization = calculate_categorization(sun)
    for parent_category, child_categories in categorization.items():
        print(f"{parent_category}: {len(child_categories)} childs --> {child_categories}")
