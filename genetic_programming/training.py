import numpy as np
import pandas as pd
import json
import random
import os

feature_names = ["home_num - away_num",
                 "home_pieces_diff - away_pieces_diff",
                 "home_stacks",
                 "turn * home_min_dist",
                 "max_damage",
                 "home_threatening",
                 "home_board_score"]


def main():
    """"""
    original = os.getcwd()

    iteration = 21

    while True:
        survive(100, iteration)

        os.chdir(original)

        evolve(iteration, select=0.3, stragglers=0.1, mutate=0.05, variance=50)

        with open("iterations.json") as file:
            scores = json.load(file)

        with open("score.json") as file:
            data = json.load(file)

        scores[iteration] = data

        with open("iterations.json", 'w') as file:
            json.dump(scores, file)

        with open("score.json", 'w') as file:
            reset = {"white": 0, "black": 0, "draw": 0}
            json.dump(reset, file)

        iteration += 1


def initialise_weights(n):

    df = pd.DataFrame(columns=feature_names)

    for i in range(n):
        weights = {}
        for j in range(len(feature_names)):
            weights[feature_names[j]] = random.uniform(-100, 100)

        df = df.append(weights, ignore_index=True)

    score = np.zeros(n)
    games = np.zeros(n)

    score_df = pd.DataFrame({"Score": score, "Games": games})

    df = score_df.merge(df, left_index=True, right_index=True)

    df.to_csv("weights.csv")


def evolve(iteration, select, stragglers, mutate, variance):

    df = pd.read_csv("weights.csv", header=0, index_col=0)
    n = len(df)
    df.to_csv("weights{}.csv".format(iteration))

    # Retain highest winning percentage
    df["accuracy"] = df["Score"] / df["Games"]
    df = df.sort_values(by="accuracy", ascending=False)
    survived = df.head(round(select * n))

    features = survived.loc[:, feature_names + ["accuracy"]]

    # Introduce stragglers
    plane_size = max(1, round(stragglers * n)) + len(features)
    plane_has_landed = False

    while not plane_has_landed:
        straggler = df.sample(1).loc[:, feature_names + ["accuracy"]]
        feature_index = features.index
        if straggler["accuracy"] <= 0:
            continue

        if len(straggler.index.difference(feature_index)) == 0:
            features = features.append(straggler)

        if len(features) == plane_size:
            plane_has_landed = True

    new_pop = features.reset_index(drop=True)

    children = pd.DataFrame(columns=feature_names)
    while (len(new_pop) + len(children)) < n:
        parents = new_pop.sample(n=2)
        weight = parents["accuracy"]

        child = {}
        for j in range(len(feature_names)):
            feature = feature_names[j]
            q3 = new_pop[feature].quantile(0.75)
            q1 = new_pop[feature].quantile(0.25)
            IQR = q3-q1
            traits = parents[feature]

            child[feature] = traits.dot(weight) + np.random.normal(scale=variance)

        children = children.append(child, ignore_index=True)

    new_pop = new_pop.append(children, ignore_index=True)

    new_pop = new_pop.drop(columns=["accuracy"])

    for i in range(len(new_pop)):
        for j in range(len(new_pop.columns)):

            if random.uniform(0, 1) < mutate:
                feature = feature_names[j]
                q3 = new_pop[feature].quantile(0.75)
                q1 = new_pop[feature].quantile(0.25)
                IQR = q3 - q1
                new_pop.iloc[i, j] += np.random.normal(scale=1.5*variance)

    score = np.zeros(n)
    games = np.zeros(n)

    score_df = pd.DataFrame({"Score": score, "Games": games})

    new_pop = score_df.merge(new_pop, left_index=True, right_index=True)

    new_pop.to_csv("weights.csv")


def simulate_games():
    os.system("python3 -m referee -c -t 90 MCTS_op MCTS_op")


def survive(n, iteration):

    os.chdir("..")
    for i in range(n):
        print("Iteration: {}, Simulation: {}".format(iteration, i + 1))

        with open("genetic_programming/score.json") as file:
            data = json.load(file)
        print("Black: {}, White: {}, Draw: {}".format(data["black"], data["white"], data["draw"]))

        simulate_games()


if __name__ == '__main__':
    main()