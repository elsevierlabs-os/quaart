import argparse

parser = argparse.ArgumentParser(description='Takes csv prefix for input and output')
parser.add_argument('-p','--prefix', help='CSV Prefix for in and out',required=True)

# Example, if you run experiment2 with the command ./experiment2.sh figer_results,
# You should now run the command python finalResultsFiger.py -p hands_results

import json
import pandas as pd
args = parser.parse_args()
test_df = pd.read_csv(args.prefix + ".csv", index_col=0)
test_df["dev_test"] = test_df.apply (lambda x: x["model name"].split("_")[2], axis = 1)
test_df["model name"] = test_df.apply (lambda x: x["model name"].split(":")[0], axis = 1)
test_df["Model Number"] =  test_df["model name"].str.extract('_([0-9][abc])_')#.astype(float)

models = list(set(test_df["model name"]))

counts = {}
for i in range(1,10):
    for l in ["a", "b", "c"]:
        with open("data/output/fqa_train/fqa_train_gold_" + str(i) + l + ".json") as json_file:
            questions = json.load(json_file)
        counts[str(i)+l] = (len(questions["data"]),
                     len([q for q in questions["data"] if q["isImpossible"] == False]),
                     len(set([q["question"] for q in questions["data"] if q["isImpossible"] == False])),
                     len(set([q["context"] for q in questions["data"]])))

counts["0a"] = (0,0,0,0)

count_df = pd.DataFrame.from_dict(counts, orient='index')
count_df['Model Number'] =count_df.index
count_df = count_df.rename(columns={0: 'Question Count', 1: 'Possible Count', 2: 'Possible Types', 3: 'Passage Count'})

top_per_model = pd.DataFrame(columns=test_df.columns)
for model in models:
    tmp_df = test_df.loc[test_df["model name"] == model]
    top_df = tmp_df[tmp_df["connl f1"] == tmp_df["connl f1"].max()]
    top_per_model = top_per_model.append(top_df, ignore_index=True)

dev_thresholds = top_per_model.loc[top_per_model["dev_test"] == "fqadev112"][["model name", "thesholds"]]
dev_thresholds["Model Number"] = dev_thresholds["model name"].str.extract('_([0-9][abc])_')#.astype(float)

withDevThresh = test_df.merge(dev_thresholds.rename(columns={"thesholds": 'dev thresholds'}), on=["Model Number"])

top_per_model_test = withDevThresh.loc[withDevThresh["dev_test"] == "fqatest112"].loc[withDevThresh["thesholds"] == withDevThresh["dev thresholds"]]
top_with_counts = top_per_model_test.merge(count_df, on=["Model Number"])

print("Figer Connl F1 Max: " + str(top_with_counts.loc[top_with_counts["EvalSet"] == "figer"]["connl f1"].max()))
print("Figer All Micro F1 Max: " + str(top_with_counts.loc[top_with_counts["EvalSet"] == "figer"]["all micro f1"].max()))
print("Figer All Macro F1 Max: " + str(top_with_counts.loc[top_with_counts["EvalSet"] == "figer"]["all macro f1"].max()))

top_with_counts.to_csv(args.prefix+"_best.csv")