"""
Gets labels and scores from vendors for all the images in the datasets
"""
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from typing import List, Tuple

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

path = os.path.abspath(__file__ + "/../")
dir_path = os.path.dirname(path)
sys.path.insert(0, dir_path)

from labelling.settings import IMAGES_PER_CATEGORY, INDOOR_PATH, LABELS_PER_IMAGE, PLACES_PATH, RESULTS_PATH, SUN_PATH, VENDORS
from labelling.vendors.amazon import AmazonLabelsGatherer
from labelling.vendors.azure import AzureLabelsGatherer
from labelling.vendors.clarifai import ClarifaiLabelsGatherer
from labelling.vendors.google import GoogleLabelsGatherer
from labelling.vendors.watson import WatsonLabelsGatherer


def restructure_places(path: Path):
    """
    Modify Places structure

    All the images are moved to a dir named as their category ground truth
    """

    # Create dictionary with categories numbers and names
    categories_names = {}
    with open(path / "categories_places365.txt", 'r') as fp:
        for line in fp:
            line_split = line.split(" ")
            category_number = line_split[1].replace("\n", "")
            category_name = line_split[0][3:].split("/")[0]
            categories_names[category_number] = category_name
    (path / "categories_places365.txt").unlink()

    # Move images to their category folder
    with open(path / "places365_val.txt", 'r') as fp:
        print("Organizing places images by categories...")
        for line in tqdm(fp, total=len(os.listdir(path))):
            line_split = line.split(" ")
            filename = line_split[0]
            category_number = line_split[1].replace("\n", "")
            category = categories_names[category_number]
            (path / category).mkdir(exist_ok=True)
            (path / filename).replace(path / category / filename)
    (path / "places365_val.txt").unlink()
    print("Restructured Places.")


def restructure_sun(path: Path):
    """
    Modify SUN structure

    All the images are moved to a dir named as their category ground truth
    """

    # Move every category folder outside its initial letter folder
    for letter_folder in path.iterdir():
        if letter_folder.is_dir():
            for category_folder in letter_folder.iterdir():
                if category_folder.is_dir():
                    (path / category_folder.name).mkdir(exist_ok=True)
                    category_folder.rename(path / category_folder.name)
            letter_folder.rmdir()
    print("Restructured SUN.")


def reduce_dataset(path: Path, n: int):
    """
    Reduce the dataset in path to n images per category
    """

    for category in path.iterdir():
        if category.is_dir():
            images = list(category.iterdir())
            for image in images[n:]:
                image.unlink()
        else:
            category.unlink()
    print(f"Dataset in {path} reduced to {n} images per category.")


def get_labels(dataset_path: Path, n: int, vendor: str, csv_path: Path):
    """
    Generates a csv with the labels and scores provided by a vendor for all the images in a dataset.
    Columns: imageName, groundTruth, vendorLabels and vendorScores

    Args:
        dataset_path: Path to the dataset
        n: maximum number of labels and scores per image
        vendor: Vendor to obtain labels from
        csv_path: Path to store the csv in
    """

    # Create a new dataframe
    df = pd.DataFrame(
        columns=[
            "imageName",
            "groundTruth",
            f"{vendor}Labels",
            f"{vendor}Scores"
        ]
    )

    # Add vendor labels and scores for each image
    for category in dataset_path.iterdir():
        for image in category.iterdir():
            try:
                labels, scores = eval(f"{vendor.capitalize()}LabelsGatherer")().get_labels(image, n)
                df.loc[len(df)] = [image.stem, category.stem, labels, scores]
                time.sleep(1)   # Some vendors limit query frequency in their free plans
            except Exception as ex:
                print(f"Failed to get labels from {vendor}: {ex}")

    # Save dataframe to disk
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(str(csv_path), index=False)


def merge_vendors_csvs(path: Path, vendors: list, dataset: str):
    """
    Merge the vendor csvs into a single one

    Args:
        path: path containing the csvs
        vendors: vendors to merge
        dataset: dataset to merge csv from
    """

    # Generate new csv
    df = pd.read_csv(path / f"{dataset}_{vendors[0]}.csv")
    for vendor in vendors[1:]:
        vendor_df = pd.read_csv(path / f"{dataset}_{vendor}.csv")
        df = pd.merge(df, vendor_df, on=["imageName", "groundTruth"], how="inner")
    df.to_csv(path / f"{dataset}.csv", index=False)

    # Delete old csvs
    for vendor in vendors:
        (path / f"{dataset}_{vendor}.csv").unlink()


def calculate_labels_per_image(path: Path, vendors: list):
    """
    Calculate the average number of labels per image for each vendor

    Args:
        path: path containing the datasets csvs
        vendors: vendors to calculate
    """

    # Merge dataset csvs
    df = pd.DataFrame()
    for dataset_path in path.iterdir():
        dataset_csv = pd.read_csv(dataset_path)
        df = pd.concat([df, dataset_csv])

    # Calculate average number of labels per image for each vendor
    for vendor in vendors:
        df[f"{vendor}LabelsCount"] = df[f"{vendor}Labels"].map(lambda x: str(x).count(",")+1)
        print(f"{vendor} average labels/image: {df[f'{vendor}LabelsCount'].mean()}")


def calculate_distinct_labels_per_vendor_and_dataset(path: Path, vendors: list):
    """
    Calculate how many distinct labels each vendor generates for each dataset

    Args:
        path: path containing the datasets csvs
        vendors: vendors to calculate
    """

    labels_sets = {vendor: set() for vendor in vendors}
    for dataset_path in path.iterdir():
        df = pd.read_csv(dataset_path)
        for _, row in df.iterrows():
            for vendor in vendors:
                try:
                    for label in eval(str(row[f"{vendor}Labels"])):
                        labels_sets[vendor].add(label)
                except Exception:   # Found a NaN
                    pass
        for vendor in vendors:
            print(f"Distinct labels of vendor {vendor} for dataset {dataset_path.stem}: {len(labels_sets[vendor])}")


def plot_vendors_score_distribution(path: Path):
    """
    Plot the labels score distribution for all vendors
    """

    # Merge dataset csvs
    df = pd.DataFrame()
    for dataset_path in path.iterdir():
        dataset_df = pd.read_csv(dataset_path)
        df = pd.concat([df, dataset_df])

    # For each image and vendor, register all the scores and the maximum one
    vendors_scores = {vendor: [] for vendor in VENDORS}
    vendors_max_scores = {vendor: [] for vendor in VENDORS}
    for image in df.iterrows():
        for vendor in VENDORS:
            try:
                image_scores = eval(image[1][f"{vendor}Scores"])
                if len(image_scores) > 0:

                    # Register all the scores
                    if image_scores[0] > 1:
                        image_scores = [score/100 for score in image_scores]
                    vendors_scores[vendor] += image_scores

                    # Register just the max score
                    vendors_max_scores[vendor].append(max(image_scores))
            except:   # Found a NaN
                pass

    for vendor in VENDORS:

        # Calculate percentiles
        scores_arr = np.array(vendors_scores[vendor])
        max_scores_arr = np.array(vendors_max_scores[vendor])
        for q in [20, 40, 60, 70, 80, 90]:
            print(f"{vendor}: all scores {q} percentile: {np.percentile(scores_arr, q)}")
        for q in [20, 40, 60, 70, 80, 90]:
            print(f"{vendor}: max scores {q} percentile: {np.percentile(max_scores_arr, q)}")

        # Plot histogram
        print(f"------- {vendor} scores distribution -------")
        plt.hist(vendors_scores[vendor], bins=100)
        plt.hist(vendors_max_scores[vendor], bins=100)
        plt.xlim((0,1))
        plt.show()

        # Normalize scores
        max_score = max(vendors_scores[vendor])
        min_score = min(vendors_scores[vendor])
        normalized_scores = [(score-min_score)/(max_score-min_score) for score in vendors_scores[vendor]]
        normalized_max_scores = [(score-min_score)/(max_score-min_score) for score in vendors_max_scores[vendor]]

        # Plot normalized histogram
        print(f"------- {vendor} normalized scores distribution -------")
        plt.hist(normalized_scores, bins=100)
        plt.hist(normalized_max_scores, bins=100)
        plt.xlim((0,1))
        plt.show()


if __name__ == "__main__":

    # Store Indoor labels
    reduce_dataset(INDOOR_PATH, IMAGES_PER_CATEGORY)
    for vendor in VENDORS:
        csv_path = RESULTS_PATH / f"indoor_{vendor}.csv"
        thread = Thread(target=get_labels, args=[INDOOR_PATH, LABELS_PER_IMAGE, vendor, csv_path])
        thread.start()
    merge_vendors_csvs(RESULTS_PATH, VENDORS, "indoor")
    
    # Store Places labels
    restructure_places(PLACES_PATH)
    reduce_dataset(PLACES_PATH, IMAGES_PER_CATEGORY)
    for vendor in VENDORS:
        csv_path = RESULTS_PATH / f"places_{vendor}.csv"
        thread = Thread(target=get_labels, args=[PLACES_PATH, LABELS_PER_IMAGE, vendor, csv_path])
        thread.start()
    merge_vendors_csvs(RESULTS_PATH, VENDORS, "places")

    # Store SUN labels
    restructure_sun(SUN_PATH)
    reduce_dataset(SUN_PATH, IMAGES_PER_CATEGORY)
    for vendor in VENDORS:
        csv_path = RESULTS_PATH / f"sun_{vendor}.csv"
        thread = Thread(target=get_labels, args=[SUN_PATH, LABELS_PER_IMAGE, vendor, csv_path])
        thread.start()
    merge_vendors_csvs(RESULTS_PATH, VENDORS, "sun")
    
    # Extract some metrics
    calculate_labels_per_image(RESULTS_PATH, VENDORS)
    calculate_distinct_labels_per_vendor_and_dataset(RESULTS_PATH, VENDORS)
    plot_vendors_score_distribution(RESULTS_PATH)
