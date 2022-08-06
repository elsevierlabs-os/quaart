from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import utils
import json
import re
import spacy
import argparse
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_sm')
import pandas as pd
from os.path import exists

with open('config.json') as cf_file:
    config = json.loads( cf_file.read() )

with open (config["data"]["figer"]["figerText"], 'r') as f:
    figerText = f.read().splitlines()

with open (config["data"]["figer"]["figerLabel"], 'r') as f:
    figerLabel = f.read().splitlines()

parser = argparse.ArgumentParser(description='Takes model output dir and gold data')
parser.add_argument('-m','--model', help='Model output directory name',required=True)
parser.add_argument('-g','--gold', help='Gold data input questions',required=True)
parser.add_argument('-o','--output', help='CSV to create or append scores',required=True)


args = parser.parse_args()

with open(args.gold) as json_file:
    questions = json.load(json_file)

contexts = set([q["context"] for q in questions['data']])

curr_toks = ''
connl_bios = []
figer_bios = []
connl_doc = []
figer_doc = []
for label in figerLabel:
    if label == "\t" or label == "":
        if curr_toks.rstrip() in list(contexts):
            connl_bios.append(connl_doc)
            figer_bios.append(figer_doc)
        connl_doc = []
        figer_doc = []
        curr_toks = ''
    elif label.split("\t")[1] == "O":
        biof = label.split("\t")[1] #.split("-")[0] + "-MISC"
        bioc = label.split("\t")[1] #.split("-")[0] + "-MISC"
        connl_doc.append(bioc)
        figer_doc.append(biof)
        tok = label.split("\t")[0]
        curr_toks += tok+' '
    else:
        biof = label.split("\t")[1] #.split("-")[0] + "-MISC"
        bioc = label.split("\t")[1].split("-")[0] + "-MISC"
        connl_doc.append(bioc)
        figer_doc.append(biof)
        tok = label.split("\t")[0]
        curr_toks += tok+' '

qLookup = {}
for q in questions['data']:
    qLookup[q['id']] = q

with open (config["data"]["figer"]["figerClasses"], 'r') as f:
    figerClasses = f.read().splitlines()

qMapper = {}
for label in figerClasses:
    labelParts = label.split("/")
    macro = labelParts[1]
    micro = labelParts[-1].replace("_"," ")
    if macro == "person":
        qStart = "Who "
    elif macro == "location":
        qStart = "Where "
    else:
        qStart = "What "
    qMapper[qStart + "was the " + micro + "?"] = label

best_scores = {}

best = (0,0)

nbest_path = args.model + '/predict_nbest_predictions.json'
with open(nbest_path) as json_file:
    nbest = json.load(json_file)

thresholds = [.01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50]
precisions = []
recalls = []
f1_scores = []
strict_p = []
loose_micro_p = []
loose_macro_p = []
strict_r = []
loose_micro_r = []
loose_macro_r = []
strict_f1 = []
loose_micro_f1 = []
loose_macro_f1 = []
all_strict_p = []
all_loose_micro_p = []
all_loose_macro_p = []
all_strict_r = []
all_loose_micro_r = []
all_loose_macro_r = []
all_strict_f1 = []
all_loose_micro_f1 = []
all_loose_macro_f1 = []

for thresh in thresholds:
    scores = {"thresh": thresh}
    entities = {}
    dedupedEntities = {}
    connl_submit = []
    figer_submit = []
    id_list = list(set(v["title"] for k,v in qLookup.items()))
    id_list.sort(key=int)
    for para in id_list:
        success_list = []
        cleaned_list = []
        dedupers = []
        entities[para] = []

        for k,v in nbest.items():
            q = qLookup[k]
            if q["title"] == para:
                context = q["context"]
                for index, top in enumerate(v):
                    if top["probability"] >= thresh and top["offsets"] != [0,0]:
                        if (top["offsets"], q["title"],  q["question"]) not in dedupers:
                            success_list.append({"title": q["title"],
                                                 "context": q["context"],
                                                 "question": q["question"],
                                                 "offsets": top["offsets"],
                                                 "text": top["text"],
                                                 "position": index
                                                })
                            cleaned_list.append({"title": q["title"],
                                                 "context": q["context"],
                                                 "question": q["question"],
                                                 "offsets": top["offsets"],
                                                 "text": top["text"],
                                                 "position": index
                                                })
                            dedupers.append((top["offsets"], q["title"], q["question"]))
        deduped = []
        counted = {}
        keepers = []
        for item in success_list:
            if item["offsets"] not in deduped:
                deduped.append(item["offsets"])
        for item in success_list:
            containing = 0
            for subitem in deduped:
                if ((subitem[0] > item["offsets"][0] and
                    subitem[1] <= item["offsets"][1]) or
                    (subitem[0] >= item["offsets"][0] and
                    subitem[1] < item["offsets"][1])):
                    containing += 1
            if containing <= 1:
                if (item["offsets"][0], item["offsets"][1]) not in counted.keys():
                    counted[(item["offsets"][0], item["offsets"][1])] = 1
                else:
                    counted[(item["offsets"][0], item["offsets"][1])] += 1
            elif item["offsets"] in deduped:
                deduped.remove(item["offsets"])
                cleaned_list.remove(item)
            else:
                cleaned_list.remove(item)
        for item in cleaned_list:
            containing = 0
            itemStart = item["offsets"][0]
            itemEnd = item["offsets"][1]
            itemKeep = True
            for subitem in deduped:
                subStart = subitem[0]
                subEnd = subitem[1]
                if ((subStart >= itemStart and subStart <= itemEnd) or
                    itemStart >= subStart and itemStart <= subEnd):
                    if counted[(itemStart, itemEnd)] < counted[(subStart, subEnd)]:
                        itemKeep = False
                    elif itemEnd-itemStart < subEnd-subStart:
                        itemKeep = False
            if itemKeep == True:
                entities[para].append(item)
        dedupedEntities = {para: {}}
        for entity in entities[para]:
            if tuple(entity['offsets']) not in dedupedEntities[para]:
                dedupedEntities[para][tuple(entity['offsets'])] = qMapper[entity["question"]]+","
            else:
                dedupedEntities[para][tuple(entity['offsets'])] += qMapper[entity["question"]]+","
        connl_sub = []
        figer_sub = []
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S').match)
        tokens = nlp(context)

        for token in tokens:
            start = token.idx
            end = token.idx+len(token.text)
            toktyp = ("O", "")
            for tup, typ in dedupedEntities[para].items():
                #print(entity)
                estart = tup[0]
                eend = tup[1]
                if start == estart:
                    toktyp = ("B", typ)
                elif start >= estart and end <= eend:
                    toktyp = ("I", typ)
            if toktyp[0] == "B":
                #print("B" + " " + token.text + " " + toktyp[1])
                connl_sub.append("B-MISC")
                figer_sub.append("B-"+toktyp[1])
            elif toktyp[0] == "I":
                #print("I" + " " + token.text + " " + toktyp[1])
                connl_sub.append("I-MISC")
                figer_sub.append("I-"+toktyp[1])
            else:
                #print("O" + " " + token.text)
                connl_sub.append("O")
                figer_sub.append("O")
        connl_submit.append(connl_sub)
        figer_submit.append(figer_sub)
    print("## Threshold: " + str(thresh))
    f = f1_score(connl_bios, connl_submit, mode='strict', scheme=IOB2 )
    print(f)
    f1_scores.append(f)
    scores["f1"] = f
    pre = precision_score(connl_bios, connl_submit, mode='strict', scheme=IOB2)
    print(pre)
    precisions.append(pre)
    scores["precision"] = pre
    rec = recall_score(connl_bios, connl_submit, mode='strict', scheme=IOB2)
    print(rec)
    recalls.append(rec)
    scores["recall"] = rec


    #Now this bit handles the figer style scoring

    goldTypeList = []
    subTypeList = []
    bothTypeList = []
    allTypeList = []
    matches = 0
    gold = 0
    subs = 0
    for bidx, bio in enumerate(figer_bios):
        entities = []
        entTypes = {}
        entity = [-1,-1]
        for idx, tok in enumerate(bio):
            if tok[0] == "B":
                if entity != [-1,-1]:
                    entities.append(entity)
                    #entTypes[tuple(entity)] = [et for et in etype.split(',') if et != ""]
                    entTypes[tuple(entity)] = etype.split(',')
                    entity = [-1,-1]
                entity = [idx, idx]
                etype = tok.split("-")[1]
            elif tok[0] == "I":
                entity[1] = idx
            elif entity != [-1,-1]:
                entities.append(entity)
                entTypes[tuple(entity)] = etype.split(',')
                entity = [-1,-1]
        if entity != [-1,-1]:
            entities.append(entity)
            entTypes[tuple(entity)] = etype.split(',')
            entity = [-1,-1]
        sub = figer_submit[bidx]
        subEntities = []
        subEntity = [-1,-1]
        subTypes = {}
        for idx, tok in enumerate(sub):
            if tok[0] == "B":
                if subEntity != [-1,-1]:
                    subEntities.append(subEntity)
                    subTypes[tuple(subEntity)] = stype.rstrip(',').split(',')
                    subEntity = [-1,-1]
                subEntity = [idx, idx]
                stype = tok.split("-")[1]
            elif tok[0] == "I":
                subEntity[1] = idx
            elif subEntity != [-1,-1]:
                subEntities.append(subEntity)
                subTypes[tuple(subEntity)] = stype.rstrip(',').split(',')
                subEntity = [-1,-1]
        if subEntity != [-1,-1]:
            subEntities.append(subEntity)
            subTypes[tuple(subEntity)] = stype.rstrip(',').split(',')
            subEntity = [-1,-1]

        matches += len(set([tuple(e) for e in entities]).intersection(set(tuple(s) for s in subEntities)))
        matchTuples = list(set([tuple(e) for e in entities]).intersection(set(tuple(s) for s in subEntities)))
        for t in matchTuples:
            goldTypeList.append(entTypes[t])
            subTypeList.append(subTypes[t])
            bothTypeList.append((entTypes[t], subTypes[t]))
        for t in list(set([tuple(e) for e in entities])):
            if t in matchTuples:
                allTypeList.append((entTypes[t], subTypes[t]))
            else:
                allTypeList.append((entTypes[t], []))
        for t in list(set([tuple(e) for e in subEntities])):
            if t not in matchTuples:
                allTypeList.append(([], subTypes[t]))
        gold += len(entities)
        subs += len(subEntities)
    # print(len(set([tuple(e) for e in entities]).intersection(set(tuple(s) for s in subEntities))))
    print("Matches: " + str(matches))
    print("Gold: " + str(gold))
    print("Sub: " + str(subs))
    scores["subCount"] = subs
    scores["matchCount"] = matches
    if subs != 0:
        s_p, s_r, s_f = utils.strict(bothTypeList)
        print(utils.strict(bothTypeList))
        l_mac_p, l_mac_r, l_mac_f = utils.loose_macro(bothTypeList)
        print(utils.loose_macro(bothTypeList))
        l_mic_p, l_mic_r, l_mic_f = utils.loose_micro(bothTypeList)
        print(utils.loose_micro(bothTypeList))
    else:
        s_p, s_r, s_f = 0,0,0
        #print(strict(allTypeList))
        l_mac_p, l_mac_r, l_mac_f = 0,0,0
        #print(loose_macro(allTypeList))
        l_mic_p, l_mic_r, l_mic_f = 0,0,0
        #print(loose_micro(allTypeList))
    if subs != 0:
        a_s_p, a_s_r, a_s_f = utils.strict(allTypeList)
        print(utils.strict(allTypeList))
        a_l_mac_p, a_l_mac_r, a_l_mac_f = utils.loose_macro(allTypeList)
        print(utils.loose_macro(allTypeList))
        a_l_mic_p, a_l_mic_r, a_l_mic_f = utils.loose_micro(allTypeList)
        print(utils.loose_micro(allTypeList))
    else:
        a_s_p, a_s_r, a_s_f = 0,0,0
        #print(strict(allTypeList))
        a_l_mac_p, a_l_mac_r, a_l_mac_f = 0,0,0
        #print(loose_macro(allTypeList))
        a_l_mic_p, a_l_mic_r, a_l_mic_f = 0,0,0
        #print(loose_micro(allTypeList))

    strict_p.append(s_p)
    loose_micro_p.append(l_mic_p)
    loose_macro_p.append(l_mac_p)
    strict_r.append(s_r)
    loose_micro_r.append(l_mic_r)
    loose_macro_r.append(l_mac_r)
    strict_f1.append(s_f)
    loose_micro_f1.append(l_mic_f)
    loose_macro_f1.append(l_mac_f)

    scores["strict"] = {"p": s_p, "r": s_r, "f1": s_f}
    scores["loose_macro"] = {"p": l_mac_p, "r": l_mac_r, "f1": l_mac_f}
    scores["loose_micro"] = {"p": l_mic_p, "r": l_mic_r, "f1": l_mic_f}

    all_strict_p.append(a_s_p)
    all_loose_micro_p.append(a_l_mic_p)
    all_loose_macro_p.append(a_l_mac_p)
    all_strict_r.append(a_s_r)
    all_loose_micro_r.append(a_l_mic_r)
    all_loose_macro_r.append(a_l_mac_r)
    all_strict_f1.append(a_s_f)
    all_loose_micro_f1.append(a_l_mic_f)
    all_loose_macro_f1.append(a_l_mac_f)

    scores["all_strict"] = {"p": a_s_p, "r": a_s_r, "f1": a_s_f}
    scores["all_loose_macro"] = {"p": a_l_mac_p, "r": a_l_mac_r, "f1": a_l_mac_f}
    scores["all_loose_micro"] = {"p": a_l_mic_p, "r": a_l_mic_r, "f1": a_l_mic_f}


    best_scores[args.model+ ":" + str(thresh)] = scores


test_dict = {"EvalSet": ["figer" for k in best_scores.keys()],
             "model name": [k for k in best_scores.keys()],
             "thesholds":[v["thresh"] for v in best_scores.values()],
             "subCount": [v["subCount"] for v in best_scores.values()],
             "matchCount": [v["matchCount"] for v in best_scores.values()],
             "connl f1": [v["f1"] for v in best_scores.values()],
             "connl p": [v["precision"] for v in best_scores.values()],
             "connl r": [v["recall"] for v in best_scores.values()],
             "figer strict f1": [v["strict"]["f1"] for v in best_scores.values()],
             "figer strict p": [v["strict"]["p"] for v in best_scores.values()],
             "figer strict r": [v["strict"]["r"] for v in best_scores.values()],
             "macro f1": [v["loose_macro"]["f1"] for v in best_scores.values()],
             "macro p": [v["loose_macro"]["p"] for v in best_scores.values()],
             "macro r": [v["loose_macro"]["r"] for v in best_scores.values()],
             "micro f1": [v["loose_micro"]["f1"] for v in best_scores.values()],
             "micro p": [v["loose_micro"]["p"] for v in best_scores.values()],
             "micro r": [v["loose_micro"]["r"] for v in best_scores.values()],
             "all figer strict f1": [v["all_strict"]["f1"] for v in best_scores.values()],
             "all figer strict p": [v["all_strict"]["p"] for v in best_scores.values()],
             "all figer strict r": [v["all_strict"]["r"] for v in best_scores.values()],
             "all macro f1": [v["all_loose_macro"]["f1"] for v in best_scores.values()],
             "all macro p": [v["all_loose_macro"]["p"] for v in best_scores.values()],
             "all macro r": [v["all_loose_macro"]["r"] for v in best_scores.values()],
             "all micro f1": [v["all_loose_micro"]["f1"] for v in best_scores.values()],
             "all micro p": [v["all_loose_micro"]["p"] for v in best_scores.values()],
             "all micro r": [v["all_loose_micro"]["r"] for v in best_scores.values()]}

test_df = pd.DataFrame(test_dict)

if exists(args.output):
    new_df = pd.read_csv(args.output, index_col=0)
    new_df = new_df.append(test_df, ignore_index=True)
else:
    new_df = test_df
new_df.to_csv(args.output)
