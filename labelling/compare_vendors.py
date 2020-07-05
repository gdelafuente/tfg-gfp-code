"""
Comparison of labels obtained from all vendors
"""
import os
import sys
from itertools import permutations
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

path = os.path.abspath(__file__ + "/../")
dir_path = os.path.dirname(path)
sys.path.insert(0, dir_path)

from labelling.settings import RESULTS_PATH, SIMILARITY_THRESHOLD, VENDORS, LABELS_PER_IMAGE

brown_ic = wordnet_ic.ic("ic-brown.dat")


def _calculate_max_similarity(a: [wn.synset], b: [wn.synset]) -> float:
    """
    Calculate the maximum Lin Similarity between synsets in a and synsets in b
    """

    max_similarity = 0
    for one in a:
        for other in b:
            max_similarity = max(max_similarity, wn.lin_similarity(one, other, brown_ic))
    return max_similarity


def _generate_word_synsets(word: str) -> [wn.synset]:
    """
    Return all noun synsets of word
    """

    word = word.lower()  # Lowercase
    word = word.replace(" ", "_")  # Replace spaces with underscores
    if "(" in word and ")" in word:
        word = word.split("(",1)[1].split(")",1)[0]  # Select text between parentheses
    return wn.synsets(word, pos='n')


def _calculate_words_similarities(a_words: [str], b_words: [str]) -> [float]:
    """
    Calculate the Lin Similarities between words in a and words in b
    """

    # Generate all synsets
    a_words_synsets = [_generate_word_synsets(word) for word in a_words]
    b_words_synsets = [_generate_word_synsets(word) for word in b_words]

    # Calculate similarities
    similarities = []
    for a_word_synsets in a_words_synsets:
        for b_word_synsets in b_words_synsets:
            similarities.append(round(_calculate_max_similarity(a_word_synsets, b_word_synsets), 2))
    return similarities


def plot_vendors_similarity_distribution(dfs: list):
    """
    Plot the similarity (between predicted labels and ground truth) distribution for all vendors and average.
    """

    # For each vendor, plot all the similarities and the best labels' ones
    df = pd.concat(dfs, sort=False)
    vendors_similarities = {vendor: [] for vendor in VENDORS}
    vendors_best_labels_similarity = {vendor: [] for vendor in VENDORS}
    for vendor in VENDORS:
        for image in df.iterrows():

            # Register all similarities
            vendors_similarities[vendor] += _calculate_words_similarities([image[1]["groundTruth"]], eval(image[1][f"{vendor}Labels"]))

            # Register just best label's similarity
            try:
                scores = eval(image[1][f"{vendor}Scores"])
                best_label_index, _ = max(list(enumerate(scores)), key=lambda x:x[1])
                best_label = eval(image[1][f"{vendor}Labels"])[best_label_index]
                vendors_best_labels_similarity[vendor] += _calculate_words_similarities([image[1]["groundTruth"]], [best_label])
            except:
                pass

        # Plot vendor similarity histogram
        plt.hist(vendors_similarities[vendor], bins=100)
        plt.hist(vendors_best_labels_similarity[vendor], bins=100)
        plt.xlim((0,1))
        print(f"------- {vendor} similarity distribution -------")
        plt.show()

        # Plot y-limited
        plt.hist(vendors_similarities[vendor], bins=100)
        plt.hist(vendors_best_labels_similarity[vendor], bins=100)
        plt.xlim((0,1))
        plt.ylim((0,350))
        print(f"------- {vendor} limited similarity distribution -------")
        plt.show()

    # Generate Latex table with percentages of labels with 0 and 1 similarity
    print(f"\\begin{{tabular}}{{|c|c|c|c|c|}}")
    print("\\hline")
    print(f"\\multirow{{2}}{{*}}{{Vendor}} & \\multicolumn{{2}}{{c|}}{{Similaridad nula (\%)}} & \\multicolumn{{2}}{{c|}}{{Similaridad plena (\%)}} \\\\") 
    print("\\cline{2-5}")
    print(f" & Total de etiquetas & \\textit{{Score}} máximo & Total de etiquetas & \\textit{{Score}} máximo \\\\") 
    for vendor in VENDORS:
        print("\\hline")
        null_similarity = vendors_similarities[vendor].count(0.0)
        full_similarity = vendors_similarities[vendor].count(1.0)
        total = len(vendors_similarities[vendor])
        null_best_similarity = vendors_best_labels_similarity[vendor].count(0.0)
        full_best_similarity = vendors_best_labels_similarity[vendor].count(1.0)
        total_best = len(vendors_best_labels_similarity[vendor])
        print(f"{vendor} & {round(null_similarity/total*100, 2)} & {round(null_best_similarity/total_best*100, 2)} & {round(full_similarity/total*100, 2)} & {round(full_best_similarity/total_best*100, 2)}\\\\")
    print("\\hline")
    print(f"\\end{{tabular}}")


    # Plot combined similarity histogram
    combined_similarities = [similarity for similarities in vendors_similarities.values() for similarity in similarities]
    combined_best_labels_similarity = [similarity for similarities in vendors_best_labels_similarity.values() for similarity in similarities]
    plt.hist(combined_similarities, bins=100)
    plt.hist(combined_best_labels_similarity, bins=100)
    plt.xlim((0,1))
    print("------- combined similarity distribution -------")
    plt.show()

    # Plot y-limited
    plt.hist(combined_similarities, bins=100)
    plt.hist(combined_best_labels_similarity, bins=100)
    plt.ylim((0,1200))
    plt.xlim((0,1))
    print(f"------- combined limited similarity distribution -------")
    plt.show()


def plot_datasets_similarity_distribution(dfs: list):
    """
    Plot the average (of vendors) similarity (between predicted labels and ground truth) distribution for all datasets
    """

    # For each dataset, plot all similarities and the best labels' ones 
    for df_index, df in enumerate(dfs):
        similarities = []
        best_labels_similarity = []

        # For each image, register all labels and best label's similarity
        for image in df.iterrows():
            for vendor in VENDORS:

                # All similarities
                similarities += _calculate_words_similarities([image[1]["groundTruth"]], eval(image[1][f"{vendor}Labels"]))

                # Just best label's similarity
                try:
                    scores = eval(image[1][f"{vendor}Scores"])
                    best_label_index, _ = max(list(enumerate(scores)), key=lambda x:x[1])
                    best_label = eval(image[1][f"{vendor}Labels"])[best_label_index]
                    best_labels_similarity += _calculate_words_similarities([image[1]["groundTruth"]], [best_label])
                except:
                    pass

        # Generate dataset's similarity histogram
        plt.hist(similarities, bins=100)
        plt.hist(best_labels_similarity, bins=100)
        plt.xlim((0,1))
        print(f"------- {df_index} dataset similarity distribution -------")
        plt.show()

        # Plot y-limited
        plt.hist(similarities, bins=100)
        plt.hist(best_labels_similarity, bins=100)
        plt.xlim((0,1))
        plt.ylim((0,600))
        print(f"------- {df_index} dataset limited similarity distribution -------")
        plt.show()

        # Calculate percentages of labels with 0 and 1 similarity
        null_similarity = similarities.count(0.0)
        full_similarity = similarities.count(1.0)
        total = len(similarities)
        print(f"All labels: {round(null_similarity/total*100, 2)}% have null similarity and {round(full_similarity/total*100, 2)}% have full similarity")
        null_best_similarity = best_labels_similarity.count(0.0)
        full_best_similarity = best_labels_similarity.count(1.0)
        total_best = len(best_labels_similarity)
        print(f"Best labels: {round(null_best_similarity/total_best*100, 2)}% have null similarity and {round(full_best_similarity/total_best*100, 2)}% have full similarity")


def calculate_vendors_coincidence(dfs: list):
    """
    Calculate the average similarity between the best labels generated by every pair of vendors

    Generate a LaTex table
    """

    # Generate unique combinations of two vendors
    pairs = []
    for index, vendor in enumerate(VENDORS):
        for other_vendor in VENDORS[:index:]:
            pairs.append({vendor, other_vendor})

    # Create an base dataframe with vendors pairs
    results = pd.DataFrame()
    results["vendor1"] = [a for (a, b) in pairs]
    results["vendor2"] = [b for (a, b) in pairs]

    # Calculate average similarity for each dataset and vendors pair
    for df_index, df in enumerate(dfs):
        for (a, b) in pairs:
            best_labels_similarity = []

            # Register best labels similarity
            for image in df.iterrows():
                try:
                    a_scores = eval(image[1][f"{a}Scores"])
                    a_best_label_index, _ = max(list(enumerate(a_scores)), key=lambda x:x[1])
                    a_best_label = eval(image[1][f"{a}Labels"])[a_best_label_index]
                    b_scores = eval(image[1][f"{b}Scores"])
                    b_best_label_index, _ = max(list(enumerate(b_scores)), key=lambda x:x[1])
                    b_best_label = eval(image[1][f"{b}Labels"])[b_best_label_index]
                    best_labels_similarity += _calculate_words_similarities([a_best_label], [b_best_label])
                except:
                    pass

            # Calculate average similarity and insert to dataframe
            avg_similarity = sum(best_labels_similarity)/len(best_labels_similarity)*100
            results.loc[(results['vendor1'] == a) & (results['vendor2'] == b), str(df_index)] = avg_similarity

    # Calculate average similarities
    results['avg'] = results[[str(df_index) for df_index, _ in enumerate(dfs)]].mean(axis=1)
    results.loc['avg'] = results.mean()

    # Print table
    results = results.round(2)
    print("------- vendors coincidence -------")
    print(results.to_latex(index=False, column_format='|c|', decimal=','))


def calculate_extreme_categories(dfs: list, n: int):
    """
    Calculate the results of the n categories with highest and lowest similarity between their labels with their ground truth

    Generate a LaTex table per image per dataset
    """

    for df_index, df in enumerate(dfs):

        # Calculate n categories with extreme ground truth similarity
        for vendor in VENDORS:
            df[f"{vendor}Similarities"] = df.apply(lambda x: _calculate_words_similarities([x["groundTruth"]], eval(x[f"{vendor}Labels"])), axis=1)
            df[f"{vendor}TotalSimilarity"] = df[f"{vendor}Similarities"].apply(lambda x: sum(x))
        df["totalSimilarity"] = df[[f"{vendor}TotalSimilarity" for vendor in VENDORS]].sum(axis=1)
        df = df.sort_values("totalSimilarity").iloc[[n-1, -n]]

        for image in df.iterrows():

            # Separate output
            print()
            print()
            print(f"------- {df_index} dataset {image[1]['imageName']} -------")
            print()
            print()

            # Generate LaTex table
            print(f"\\begin{{tabular}}{{{'|c'*(LABELS_PER_IMAGE+2)}|}}")
            print("\\hline")
            for vendor in VENDORS:
                labels_str = ""
                for label in eval(image[1][f"{vendor}Labels"]):
                    labels_str += f"\\textit{{{label.lower()}}} & "
                if labels_str.count("&") < LABELS_PER_IMAGE:
                    labels_str += "& " * (LABELS_PER_IMAGE - labels_str.count("&"))
                print(f"\\multirow{{3}}{{*}}{{\\rotatebox[origin=c]{{90}}{str(vendor).capitalize()}}} & Etiqueta & {labels_str[:-3]} \\\\")
                print("\\cline{2-7}")
                scores_str = ""
                for score in eval(image[1][f"{vendor}Scores"]):
                    if score < 1:
                        score *= 100 
                    scores_str += f"\\textit{{{str(round(score, 2)).replace('.', ',')}}} & "
                if scores_str.count("&") < LABELS_PER_IMAGE:
                    scores_str += "& " * (LABELS_PER_IMAGE - scores_str.count("&"))
                print(f"& Score (\\%) & {scores_str[:-3]} \\\\")
                print("\\cline{2-7}")
                similarities_str = ""
                for similarity in image[1][f"{vendor}Similarities"]:
                    similarities_str += f"\\textit{{{str(round(similarity*100, 2)).replace('.', ',')}}} & "
                if similarities_str.count("&") < LABELS_PER_IMAGE:
                    similarities_str += "& " * (LABELS_PER_IMAGE - similarities_str.count("&"))
                print(f"& Similaridad (\\%) & {similarities_str[:-3]} \\\\")
                print("\\hline")
                print("\\hline")
            print(f"\\end{{tabular}}")


def calculate_vendors_results(dfs: list):
    """
    Calculate the average similarity (between max score label and ground truth) for each vendor

    Generate a LaTex table per image per dataset
    """

    # Create an base dataframe
    results = pd.DataFrame()
    results["vendor"] = VENDORS

    # Calculate average similarity for each dataset and vendor
    for df_index, df in enumerate(dfs):
        for vendor in VENDORS:
            best_labels_similarity = []

            # Register best labels similarity
            for image in df.iterrows():
                try:
                    scores = eval(image[1][f"{vendor}Scores"])
                    best_label_index, _ = max(list(enumerate(scores)), key=lambda x:x[1])
                    best_label = eval(image[1][f"{vendor}Labels"])[best_label_index]
                    best_labels_similarity += _calculate_words_similarities([best_label], [image[1]["groundTruth"]])
                except:
                    pass

            # Calculate average similarity and insert to dataframe
            avg_similarity = sum(best_labels_similarity)/len(best_labels_similarity)*100
            results.loc[results['vendor'] == vendor, str(df_index)] = avg_similarity

    # Calculate average similarities
    results['avg'] = results[[str(df_index) for df_index, _ in enumerate(dfs)]].mean(axis=1)
    results.loc['avg'] = results.mean()

    # Print table
    results = results.round(2)
    print("------- vendors results -------")
    print(results.to_latex(index=False, column_format='|c|', decimal=','))



if __name__ == "__main__":

    # Load datasets results
    indoor = pd.read_csv(RESULTS_PATH / "indoor.csv")
    places = pd.read_csv(RESULTS_PATH / "places.csv")
    sun = pd.read_csv(RESULTS_PATH / "sun.csv")

    # Select images labelled by all vendors
    indoor = indoor.dropna(subset=["amazonLabels", "azureLabels", "clarifaiLabels", "googleLabels", "watsonLabels"])
    places = places.dropna(subset=["amazonLabels", "azureLabels", "clarifaiLabels", "googleLabels", "watsonLabels"])
    sun = sun.dropna(subset=["amazonLabels", "azureLabels", "clarifaiLabels", "googleLabels", "watsonLabels"])

    # Plot results
    plot_vendors_similarity_distribution([indoor, places, sun])
    plot_datasets_similarity_distribution([indoor, places, sun])

    # Calculate results
    calculate_vendors_coincidence([indoor, places, sun])
    calculate_extreme_categories([indoor, places, sun], n=1)
    calculate_vendors_results([indoor, places, sun])
