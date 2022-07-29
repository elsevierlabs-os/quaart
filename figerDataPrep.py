import json
from sklearn.model_selection import train_test_split
import random
from random import randint
import numpy

def partition (list_in, n, s):
    random.Random(s).shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

with open('config.json') as cf_file:
    config = json.loads( cf_file.read() )

with open (config["data"]["figer"]["figerText"], 'r') as f:
    figerText = f.read().splitlines()

with open(config["data"]["figer"]["figerLabel"], 'r') as f:
    figerLabel = f.read().splitlines()
print("Processing FIGER data")
print("Bulding main figer QA data files")
# Okay, this is the main bit that's collecting the entities
# and now we're also building an entity lookup table
tmpLabel = figerLabel.copy()
textIndex = 0
tokenCount = 0
entities = []
entlookup = {}
entity = ""
eStart = 0
eEnd = 0
counter = 0
# for t in [figerText[0]]:
for t in figerText:
    entlookup[counter] = []
    start = 0
    end = 0
    # print(t)
    # print(t.split(" ")[0])
    # print(t.split(" "))
    tokens = t.split(" ")
    count = len(tokens)
    # print(count)
    tokenCount += count
    labelsThis = tmpLabel[0:count]
    del tmpLabel[:count + 1]

    # labelsThis = tmpLabel.pop(count)
    # print(labelsThis[0].split("\t")[0])
    # if ((labelsThis[0].split("\t")[0] != tokens[0]) and
    #         (labelsThis[-1].split("\t")[0] != tokens[-1])):
    #     print(t)
    for label in labelsThis:
        word = label.split("\t")[0]
        tag = label.split("\t")[1]
        start = end
        end = end + len(word) + 1
        # print(t[start:end])
        if tag[0] == "B":
            # LOL First time I only added this on an "0"
            # Needs to also happen on a "B"
            if entity != "":
                entities.append((counter, entity, types, eStart, eEnd))
                entlookup[counter].append((counter, entity, types, eStart, eEnd))
            tag, types = tag.split("-")
            # print(word)
            # print(t[start:end])
            entity = word
            # print(entity)
            eStart = start
            eEnd = end
        elif tag[0] == "I":
            tag, types = tag.split("-")
            # print(word)
            # print(t[start:end])
            entity = entity + " " + word
            # print(entity)
            eEnd = end
        else:
            # print(entity)
            if entity != "":
                entities.append((counter, entity, types, eStart, eEnd))
                entlookup[counter].append((counter, entity, types, eStart, eEnd))
                # print(entity)
                entity = ""
                # print(t[eStart:eEnd])
    counter += 1

# print(tokenCount)
# print(len(figerText[1].split(" ")))
# for entity in entities:
# print(entity)
# print(t[entity[2]:entity[3]])

with open (config["data"]["figer"]["figerClasses"], 'r') as f:
    figerClasses = f.read().splitlines()

figerQA = {"version": "v2.0", "data": []}
figerGold = {"version": "v2.0", "data": []}

sentCounter = 0
questionCounter = 0
for sent in figerText:
    ents = entlookup[sentCounter]
    # print(ents)
    entsByLabel = {}
    for ent in ents:
        entlabels = ent[2].split(',')
        for entlabel in entlabels:
            l = entlabel.split("/")[-1].replace("_", " ")
            if l not in entsByLabel.keys():
                entsByLabel[l] = [{"text": ent[1], "start": ent[3]}]
            else:
                entsByLabel[l].append({"text": ent[1], "start": ent[3]})

    #print(entsByLabel)
    title = str(sentCounter)
    paragraphs = []
    sentCounter += 1
    for label in figerClasses:
        questionCounter += 1
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
        figerQA["data"].append({"id": str(questionCounter),
                                "title": str(sentCounter),
                                "context": sent,
                                "question": qStart + "was the " + micro + "?"})
        paragraphs = []
        qas = []
        answers = []
        # print(micro)
        # print(entsByLabel.keys())
        if micro in entsByLabel.keys():
            #print(micro)
            #print(entsByLabel.keys())
            for ent in entsByLabel[micro]:
                #print(ent)
                answers.append(ent)
            #print(answers)
            qas.append({"question": qStart + "was the " + micro + "?",
                        "id": str(questionCounter),
                        "answers": answers,
                        "is_impossible": False
                        })
        #             figerGold["data"].append({"id": str(questionCounter),
        #                    "title": str(sentCounter),
        #                    "context": sent,
        #                    "question": qStart + "was the " + micro + "?",
        #                    "answers": answers,
        #                    "is_impossible": False})
        else:
            qas.append({"question": qStart + "was the " + micro + "?",
                        "id": str(questionCounter),
                        "answers": answers,
                        "is_impossible": True
                        })
        paragraphs.append({"qas": qas,
                           "context": sent})
        figerGold["data"].append({"title": str(sentCounter),
                                  "paragraphs": paragraphs
                                  })
        #
        #     "text": "10th and 11th centuries",
        #      "answer_start": 94
        #
        # 2 experiments to try:
        # What if we use what for everything?
        # What about event? Should that have all of When, Where, What?
with open(config["output"]["figer"]["figerQA"], 'w') as f:
    json.dump(figerQA, f)
with open(config["output"]["figer"]["figerGold"], 'w') as f:
    json.dump(figerGold, f)

contexts = []
for q in figerQA['data']:
    if q["context"] not in contexts:
        contexts.append(q["context"])

print("Building training splits and primary train/dev/test files")
figer_c_tmp, figer_c_test = train_test_split(contexts,test_size=0.75, random_state=42)
figer_c_train, figer_c_val = train_test_split(figer_c_tmp,test_size=0.5, random_state=42)

figer_q_tmp, figer_q_test = train_test_split(figerQA['data'],test_size=0.75, random_state=42)
figer_q_train, figer_q_val = train_test_split(figer_q_tmp,test_size=0.5, random_state=42)

figer_cq_train = [q for q in figerQA['data'] if q["context"] in figer_c_train]
figer_cq_val = [q for q in figerQA['data'] if q["context"] in figer_c_val]
figer_cq_test = [q for q in figerQA['data'] if q["context"] in figer_c_test]

fqa_train = {"version": "v2.0", "data": figer_cq_train}
fqa_dev = {"version": "v2.0", "data": figer_cq_val}
fqa_test = {"version": "v2.0", "data": figer_cq_test}

with open(config["output"]["figer"]["fqa_train"], 'w') as f:
    json.dump(fqa_train, f)
with open(config["output"]["figer"]["fqa_dev"], 'w') as f:
    json.dump(fqa_dev, f)
with open(config["output"]["figer"]["fqa_test"], 'w') as f:
    json.dump(fqa_test, f)

### Ugh, this is a mess. Now I have to flatten the nested, and then dry the wets then wet the dries.

print("Building shuffled training data for experiment 2")
# do the same for train:
trainIds = [q['id'] for q in fqa_train["data"]]
fqa_train_gold = {'version': 'v2.0', 'data': []}
idCount = 0
for i in trainIds:
#for i in [devIds[0]]:
    #print(i)
    idCount+=1
    qq = [q for q in fqa_train["data"] if q["id"] == i]
    qg = [q for q in figerGold['data'] if q['paragraphs'][0]['qas'][0]['id'] == i]
    #print(qq)
    #print(qg)
    for qset in qg:
        for para in qset["paragraphs"]:
            for qa in para["qas"]:
                if qa["answers"] == []:
                    ans = {'answer_start': [], 'text': []}
                    imp = True
                else:
                    #[{'text': 'Department of Chemistry', 'start': 34}]
                    ans = {'answer_start': [qa["answers"][0]['start']],
                           'text': [qa["answers"][0]['text']]}
                    imp = False
                item = {'id': qa['id'],
                        'title': qset['title'],
                        'context': para['context'],
                        'question': qa['question'],
                        'answers': ans,
                        'isImpossible': imp
                        }
    #fqa_train_gold["data"].append(qg[0])
    fqa_train_gold["data"].append(item)
#print(idCount)

if config["data"]["figer"]["newShuffles"] == False:
    #print("false")
    figerShufflesFile = config["data"]["figer"]["figerShuffles"]
    with open(figerShufflesFile) as fs_file:
        figerShuffles = json.loads(fs_file.read())
    for k,v in figerShuffles.items():
        filename = config["output"]["figer"]["fqa_exp"] + "fqa_train_gold_"+k+".json"
        gold_array = []
        #print(v)
        for title in v:
            for q in fqa_train_gold["data"]:
                if q["title"] == title:
                # if q["title"] in v:
                    #print(q["title"])
                    gold_array.append(q)
        fqa_train_increment = {'version': 'v2.0', 'data': gold_array}
        with open(filename, 'w') as f:
            json.dump(fqa_train_increment, f)

elif config["data"]["figer"]["newShuffles"] == True:
    randseed = config["data"]["figer"]["randseed"]
    trainTitles = list(set([q["title"] for q in fqa_train_gold["data"]]))
    trainTitles.sort()
    # trainTitles = []
    # for q in fqa_train_gold["data"]:
    #     if q["title"] not in trainTitles:
    #         trainTitles.append(q["title"])
    random.seed(randseed)
    #fiveNines = []
    for letter in ["a", "b", "c", "d", "e"]:
        rand = randint(1, 100)
        #print(rand)
        nine_context_sets = partition(trainTitles, 9, rand)
        #print(str(nine_context_sets[0][0]) + "-" + str(nine_context_sets[-1][-1]))
        counter = 1
        gold_array = []
        #fiveNines.append(nine_context_sets)
        for con_set in nine_context_sets:
            for con in con_set:
                for q in fqa_train_gold["data"]:
                    if q["title"] == con:
                        gold_array.append(q)
            # print(con_set)
            # print(len(gold_array))
            # print(43*6*counter)
            fqa_train_increment = {'version': 'v2.0', 'data': gold_array}
            filename = config["output"]["figer"]["fqa_exp"] + "fqa_train_gold_"  + str(counter) + letter + ".json"
            with open(filename, 'w') as f:
                json.dump(fqa_train_increment, f)
            counter += 1
