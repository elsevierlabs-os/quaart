import gzip
import os
import json
from sklearn.model_selection import train_test_split
import gdown

# a file

with open('config.json') as cf_file:
    config = json.loads( cf_file.read() )

with open (config["data"]["hands"]["fner_dev"], 'r') as json_file:
    fner_dev = json.load(json_file)

with open(config["data"]["hands"]["fner_test"], 'r') as json_file:
    fner_test = json.load(json_file)

with open(config["data"]["hands"]["wfb_all"], 'r') as json_file:
    wfb_all = json.load(json_file)

filenames = []
trainpath = config["data"]["hands"]["trainpath"]

print("Processing HAnDS data")
print("Building source data used by evaluation scripts")
# Build source data used by evaluation scripss
fner_texts = []
fner_labels = []
#for hands in [fner_dev[0]]:
for hands in wfb_all:
    fner_text = ""
    for idx, token in enumerate(hands["tokens"]):
        isEnt = False
        fner_text+=token+" "
        for mention in hands["mentions"]:
            if idx == mention["start"]:
                isEnt = True
                fner_labels.append(token+"\tB-"+",".join(mention["labels"]))
            elif idx > mention["start"] and idx < mention["end"]:
                isEnt = True
                fner_labels.append(token+"\tI-"+",".join(mention["labels"]))
        if isEnt == False:
            fner_labels.append(token+"\tO")
    fner_labels.append("\t")
    fner_texts.append(fner_text)

with open(config["data"]["hands"]["fner_texts"], 'w') as fp:
    for item in fner_texts:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(config["data"]["hands"]["fner_labels"], 'w') as fp:
    for item in fner_labels:
        # write each item on a new line
        fp.write("%s\n" % item)

print("Building type list from all training data -- this will take a while.")
for dirname in os.listdir(trainpath):
    for fn in os.listdir(trainpath + "/" + dirname):
        filenames.append(trainpath+"/"+dirname+"/"+fn)


allTypes = []
for fn in filenames:
    with gzip.open(fn, 'rb') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        for ent in result["links"]:
            for label in ent["labels"]:
                if label not in allTypes:
                    allTypes.append(label)

print("Generating hands fner_dev data.")
handsDev = {"version": "v2.0", "data": []}
qCount = 1
for hands in fner_dev:
    #print(hands)
    context = ""
    title = str(hands["fileid"])+":"+str(hands["pid"])+":"+str(hands["senid"])
    labels = {}
    for token in hands["tokens"]:
        context += token + " "
    for ent in hands["mentions"]:
        start = ent["start"]
        text = ent["name"]
        for label in ent["labels"]:
            labels[label] = (start, text)
    for t in allTypes:
        labelParts = t.split("/")
        macro = labelParts[1]
        micro = labelParts[-1].replace("_"," ")
        if macro == "person":
            qStart = "Who "
        elif macro == "location":
            qStart = "Where "
        else:
            qStart = "What "
        if t not in labels.keys():
            #print(t)
            handsDev["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {"answer_start": [], "text": []},
                                       "isImpossible": True})
        else:
            # Actually, gotta convert this to token number to char number
            #
            s = labels[t][0]
            charstart = 0
            toks = context.split(" ")
            for tok in toks[:s]:
                charstart += len(tok)+1
#             print(s)
#             print(charstart)
#             q["answers"]["answer_start"][0] = charstart
            handsDev["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {
                                           "answer_start": [charstart],
                                           "text": [labels[t][1]]},
                                       "isImpossible": False})

        qCount+=1

with open(config["output"]["hands"]["handsDevQA"], 'w') as f:
    json.dump(handsDev, f)

print("Generating full hands evaluation set.")
handsEvalAll = {"version": "v2.0", "data": []}
qCount = 1
for hands in wfb_all:
    context = ""
    title = str(hands["fileid"])+":"+str(hands["pid"])+":"+str(hands["senid"])
    labels = {}
    for token in hands["tokens"]:
        context += token + " "
    for ent in hands["mentions"]:
        start = ent["start"]
        text = ent["name"]
        for label in ent["labels"]:
            labels[label] = (start, text)
    for t in allTypes:
        labelParts = t.split("/")
        macro = labelParts[1]
        micro = labelParts[-1].replace("_"," ")
        if macro == "person":
            qStart = "Who "
        elif macro == "location":
            qStart = "Where "
        else:
            qStart = "What "
        if t not in labels.keys():
            #print(t)
            handsEvalAll["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {"answer_start": [], "text": []},
                                       "isImpossible": True})
        else:
            # Actually, gotta convert this to token number to char number
            #
            s = labels[t][0]
            charstart = 0
            toks = context.split(" ")
            for tok in toks[:s]:
                charstart += len(tok)+1
#             print(s)
#             print(charstart)
#             q["answers"]["answer_start"][0] = charstart
            handsEvalAll["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {
                                           "answer_start": [charstart],
                                           "text": [labels[t][1]]},
                                       "isImpossible": False})

        qCount+=1

with open(config["output"]["hands"]["handsEvalAllQA"], 'w') as f:
    json.dump(handsEvalAll, f)

print("Building shuffled training data for experiment 1")
# Build our training data cycles from a subset of the full training data.
# Start by selecting subset of the files:
randseed = config["data"]["figer"]["randseed"]

hands_f_train, hands_f_test = train_test_split(filenames,test_size=0.95, random_state=randseed)
#print(len(hands_f_train))
#print(hands_f_train[:5])

sentCount = 0
entCount = 0
pidsiddids = []
#allJson = []
import gzip
train_f_pidsiddids = []
train_f_allTypes = []
#allJson = []
import gzip
#for fn in filenames:
for fn in hands_f_train:
#for fn in [filenames[0]]:

    #print(fn)
    #with open(fn) as textfile:
    with gzip.open(fn, 'rb') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        sentCount +=1
        for ent in result["links"]:
            entCount +=1
            for label in ent["labels"]:
                if label not in train_f_allTypes:
                    train_f_allTypes.append(label)
        pidsiddid = (result["pid"], result["sid"], result["did"])
        train_f_pidsiddids.append(pidsiddid)

hands_c_train, hands_c_test = train_test_split(train_f_pidsiddids,test_size=0.9995, random_state=42)
#print(len(hands_c_train))

filecount = 0
hands_train_larger = []
for fn in hands_f_train:
    with gzip.open(fn, 'rb') as f:
        json_list = list(f)
    for json_str in json_list:
        result = json.loads(json_str)
        pidsiddid = (result["pid"], result["sid"], result["did"])
        if pidsiddid in hands_c_train:
            hands_train_larger.append(result)
    filecount += 1
    #print(filecount)
    #print(fn)
    #print(len(hands_train_larger))

handsTrainLarger = {"version": "v2.0", "data": []}
qCount = 1
for hands in hands_train_larger:
    context = ""
    title = str(hands["did"])+":"+str(hands["pid"])+":"+str(hands["sid"])
    labels = {}
    for token in hands["tokens"]:
        context += token + " "
    for ent in hands["links"]:
        start = ent["start"]
        text = ent["name"]
        for label in ent["labels"]:
            labels[label] = (start, text)
    for t in allTypes:
        labelParts = t.split("/")
        macro = labelParts[1]
        micro = labelParts[-1].replace("_"," ")
        if macro == "person":
            qStart = "Who "
        elif macro == "location":
            qStart = "Where "
        else:
            qStart = "What "
        if t not in labels.keys():
            #print(t)
            handsTrainLarger["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {"answer_start": [], "text": []},
                                       "isImpossible": True})
        else:
            # Ugh forgot to add this block the first time jeebus
            s = labels[t][0]
            charstart = 0
            toks = context.split(" ")
            for tok in toks[:s]:
                charstart += len(tok)+1
            handsTrainLarger["data"].append({"id": str(qCount),
                                       "title": title,
                                       "context": context,
                                       "question": qStart + "was the " + micro + "?",
                                       "answers": {
                                           "answer_start": [charstart],
                                           "text": [labels[t][1]]},
                                       "isImpossible": False})

        qCount+=1

if config["data"]["hands"]["newShuffles"] == False:
    #print("false")
    handsShufflesFile = config["data"]["hands"]["handsShuffles"]
    with open(handsShufflesFile) as fs_file:
        handsShuffles = json.loads(fs_file.read())
    for k,v in handsShuffles.items():
        #if k == "hands_train_gold_nines_1a.json":
            filename = config["output"]["hands"]["hands_exp"]+k
            gold_array = []
            #print(v)
            for title in v:
                #print(title)
                for q in handsTrainLarger["data"]:
                    #print(q["title"])
                    if q["title"] == title:
                    # if q["title"] in v:
                        #print(q["title"])
                        gold_array.append(q)
            if k == "hands_train_gold_nines_6a.json":
                # Gotta add extra json to 5a to get all types
                # hands_train_increment currently contains 5a
                hands5a = hands_train_increment.copy()
                hqMapper = {}
                for label in allTypes:
                    labelParts = label.split("/")
                    macro = labelParts[1]
                    micro = labelParts[-1].replace("_", " ")
                    # print(label + "-" + macro + "-" + micro)
                    if macro == "person":
                        qStart = "Who "
                    elif macro == "location":
                        qStart = "Where "
                    else:
                        qStart = "What "
                    hqMapper[qStart + "was the " + micro + "?"] = label
                train5aTypes = list(set(
                    [hqMapper[q["question"]] for q in hands5a["data"] if q["isImpossible"] == False]
                ))
                missingTypes5a = [t for t in allTypes if t not in train5aTypes]
                #len(missingTypes5a)
                extraJson5a = []
                # allJson = []
                import gzip

                for fn in hands_f_train:
                    if len(missingTypes5a) > 0:
                        with gzip.open(fn, 'rb') as f:
                            json_list = list(f)

                        for json_str in json_list:
                            result = json.loads(json_str)
                            for ent in result["links"]:
                                entCount += 1
                                for label in ent["labels"]:
                                    if label in missingTypes5a:
                                        #print(label)
                                        extraJson5a.append(result)
                                        missingTypes5a.remove(label)
                hands_train_increment = hands5a.copy()
                qCount = 100001
                for hands in extraJson5a:
                    context = ""
                    title = str(hands["did"]) + ":" + str(hands["pid"]) + ":" + str(hands["sid"])
                    labels = {}
                    for token in hands["tokens"]:
                        context += token + " "
                    for ent in hands["links"]:
                        start = ent["start"]
                        text = ent["name"]
                        for label in ent["labels"]:
                            labels[label] = (start, text)
                    for t in allTypes:
                        labelParts = t.split("/")
                        macro = labelParts[1]
                        micro = labelParts[-1].replace("_", " ")
                        if macro == "person":
                            qStart = "Who "
                        elif macro == "location":
                            qStart = "Where "
                        else:
                            qStart = "What "
                        if t not in labels.keys():
                            # print(t)
                            hands_train_increment["data"].append({"id": str(qCount),
                                                          "title": title,
                                                          "context": context,
                                                          "question": qStart + "was the " + micro + "?",
                                                          "answers": {"answer_start": [], "text": []},
                                                          "isImpossible": True})
                        else:
                            # Ugh forgot to add this block the first time jeebus
                            s = labels[t][0]
                            charstart = 0
                            toks = context.split(" ")
                            for tok in toks[:s]:
                                charstart += len(tok) + 1
                            hands_train_increment["data"].append({"id": str(qCount),
                                                          "title": title,
                                                          "context": context,
                                                          "question": qStart + "was the " + micro + "?",
                                                          "answers": {
                                                              "answer_start": [charstart],
                                                              "text": [labels[t][1]]},
                                                          "isImpossible": False})

                        qCount += 1
            else:
                hands_train_increment = {'version': 'v2.0', 'data': gold_array}
            with open(filename, 'w') as f:
                json.dump(hands_train_increment, f)
