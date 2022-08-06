import argparse

parser = argparse.ArgumentParser(description='Takes csv prefix for input and output')
parser.add_argument('-p','--prefix', help='CSV Prefix for in and out',required=True)

# Example, if you run experiment1 with the command ./experiment1.sh hands_results,
# You should now run the command python finalResultsHands.py -p hands_results

import json
import pandas as pd
args = parser.parse_args()
test_df = pd.read_csv(args.prefix + ".csv", index_col=0)
test_df["dev_test"] = test_df.apply (lambda x: x["model name"].split("_")[3], axis = 1)
test_df["model name"] = test_df.apply (lambda x: x["model name"].split(":")[0], axis = 1)
test_df["Model Number"] =  test_df["model name"].str.extract('_([0-9][abc])')
print(test_df)

models = list(set(test_df["model name"]))
print(models)

counts = {}
for i in range(1,7):
    for l in ["a"]:
        with open("data/output/hands_train/hands_train_gold_nines_" + str(i) + l + ".json") as json_file:
            questions = json.load(json_file)
#         print(i)
#         print(len(questions["data"]))
#         print(len([q for q in questions["data"] if q["isImpossible"] == False]))
#         print(len(set([q["question"] for q in questions["data"] if q["isImpossible"] == False])))
        #print(len())
        counts[str(i)+l] = (len(questions["data"]),
                     len([q for q in questions["data"] if q["isImpossible"] == False]),
                     len(set([q["question"] for q in questions["data"] if q["isImpossible"] == False])),
                     len(set([q["context"] for q in questions["data"]])))
counts["0a"] = (0,0,0,0)

count_df = pd.DataFrame.from_dict(counts, orient='index')
count_df['Model Number'] =count_df.index
count_df = count_df.rename(columns={0: 'Question Count', 1: 'Possible Count', 2: 'Possible Types', 3: 'Passage Count'})
print(count_df)

top_per_model = pd.DataFrame(columns=test_df.columns)
for model in models:
    tmp_df = test_df.loc[test_df["model name"] == model]
    top_df = tmp_df[tmp_df["connl f1"] == tmp_df["connl f1"].max()]
    top_per_model = top_per_model.append(top_df, ignore_index=True)

print(top_per_model)

dev_thresholds_figer = top_per_model.loc[top_per_model["dev_test"] == "figerdev"][["EvalSet", "model name", "thesholds"]]
dev_thresholds_hands = top_per_model.loc[top_per_model["dev_test"] == "handsdev"][["EvalSet", "model name", "thesholds"]]
dev_thresholds = dev_thresholds_figer.append(dev_thresholds_hands)
# dev_thresholds = top_per_model.loc[top_per_model["dev_test"] == "fqadev112"][["model name", "thesholds"]]

dev_thresholds["Model Number"] = dev_thresholds["model name"].str.extract('_([0-9][abc])')#.astype(float)

withDevThresh = test_df.merge(dev_thresholds.rename(columns={"thesholds": 'dev thresholds'}), on=["EvalSet", "Model Number"])
# withDevThresh = test_df.merge(dev_thresholds.rename(columns={"thesholds": 'dev thresholds'}), on=["Model Number"])

top_per_model_figer = withDevThresh.loc[withDevThresh["dev_test"] == "figerall"].loc[withDevThresh["thesholds"] == withDevThresh["dev thresholds"]]
top_per_model_hands = withDevThresh.loc[withDevThresh["dev_test"] == "handsall"].loc[withDevThresh["thesholds"] == withDevThresh["dev thresholds"]]

top_with_counts_hands = top_per_model_hands.merge(count_df, on=["Model Number"])
top_with_counts_figer = top_per_model_figer.merge(count_df, on=["Model Number"])
top_with_counts_all = top_with_counts_figer.append(top_with_counts_hands)

print("Figer Connl F1 Max: " + str(top_with_counts_all.loc[top_with_counts_all["EvalSet"] == "hands"]["connl f1"].max()))
print("Figer All Micro F1 Max: " + str(top_with_counts_all.loc[top_with_counts_all["EvalSet"] == "hands"]["all micro f1"].max()))
print("Figer All Macro F1 Max: " + str(top_with_counts_all.loc[top_with_counts_all["EvalSet"] == "hands"]["all macro f1"].max()))

top_with_counts_all.to_csv(args.prefix+"_best.csv")
