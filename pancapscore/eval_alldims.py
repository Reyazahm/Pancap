import os
import pdb
import json
import tqdm
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases
text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to('cuda:0').eval()
import argparse

# Parse input parameters
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--cand_file', type=str)
parser.add_argument('--gt_file', type=str)
parser.add_argument('--gt4pred_answer_file', type=str)
parser.add_argument('--pred4gt_answer_file', type=str)
args = parser.parse_args()
cand_file = args.cand_file
gt_file = args.gt_file
gt4pred_answer_file = args.gt4pred_answer_file
pred4gt_answer_file = args.pred4gt_answer_file

# hyperparameter for calculating entity and location metrics
ethr = 0.5
lthr = 0.5

# function to transform a string to bbox coordinates (if in wrong format, return None)
def str2loc(s):
    try:
        s = s.lstrip('[').rstrip(']')
        s = [float(ss) for ss in s.split(',')]
        if len(s) != 4:
            raise NotImplementedError
        return s
    except Exception as e:
        return None


def parse_content(instr, img_path, mode):
    inv_cnt = 0
    tot_cnt = 0                                                                     # the total number of lines
    dct = {
        'str': instr, 
        'entity': {},
        'location': {}, 
        'attribute': [],
        'relation': [],
        'global': [],
    }
    inv_dct = {img_path: 0}
    str_list = []                                                                   # list of strings for generating questions
    content_list = instr.split('\n')
    for ii, content in enumerate(content_list):
        # handle some bugs of LLM
        if content == '':
            continue
        # 
        # basic parsing starts
        try:
            idx = int(content.split('|')[0].rstrip(' '))
            content = content.replace('_', ' ')                                     # remove "_"
            cls = content.split('|')[1].lstrip(' ').split(' -->> ')[0]
            assert cls in ['entity', 'location', 'attribute', 'relation', 'global']
            if cls == 'entity':
                val = content.split('|')[1].lstrip(' ').split(' -->> ')[1].lstrip('(').rstrip(') ').split(',')
                val = [v.lstrip(' ').rstrip(' ') for v in val]
            elif cls == 'attribute':
                val = content.split('|')[1].lstrip(' ').split(' -->> ')[1].lstrip('(').rstrip(') ').split(',')
                val = [val[0].lstrip(' ').rstrip(' '), ','.join(val[1:]).lstrip('\" ').rstrip('\"')]
            elif cls == 'relation':
                val = content.split('|')[1].lstrip(' ').split(' -->> ')[1].lstrip('(').rstrip(') ').split(',')
                val = [val[0].lstrip(' ').rstrip(' '), val[1].lstrip(' ').rstrip(' '), ','.join(val[2:]).lstrip('\" ').rstrip('\"')]
            elif cls == 'global':
                val = content.split('|')[1].lstrip(' ').split(' -->> ')[1].lstrip('(').rstrip(') ').split(',')
                val = [','.join(val[0:]).lstrip('\" ').rstrip('\"')]
            elif cls == 'location':
                val = content.split('|')[1].lstrip(' ').split(' -->> ')[1].lstrip('(').rstrip(') ').split(',')
                val = [val[0].lstrip(' ').rstrip(' '), ','.join(val[1:]).lstrip('\" ').rstrip('\"')]
            else:
                raise NotImplementedError
        except Exception as e:
            continue

        # 
        # if no error during the basic parsing, then go on
        tot_cnt += 1
        if cls == 'entity':
            dct['entity']['{} {}'.format(cls, idx)] = val[-1]
            str_list.append(content+'\n')
        elif cls == 'attribute':
            if 'answer' in mode:                        # handle answers
                if val[0].startswith('entity'):
                    dct[cls].append((val[0], val[1]))
                else:   
                    inv_cnt += 1
                    inv_dct[img_path] += 1
            elif val[0].startswith('entity'):
                if val[0] in dct['entity']: 
                    if val[0] in val[1]:
                        dct[cls].append((val[0], val[1]))
                    else:
                        inv_cnt += 1
                        inv_dct[img_path] += 1
                else:                   
                    inv_cnt += 1                        # this line of code is called due to a non-existent entity (usually a location) 
                    inv_dct[img_path] += 1
            elif val[0] == 'global' or val[0] == 'image':
                dct['global'].append(val[1])
            elif 'entity {}'.format(val[0]) in dct['entity']:
                dct[cls].append(('entity {}'.format(val[0]), val[1]))
            else:
                inv_cnt += 1                            # usually, this line of code will not be called 
                inv_dct[img_path] += 1

        elif cls == 'relation':
            if 'answer' in mode:                        # handle answers
                if len(val) == 3 and val[0].startswith('entity') and val[1].startswith('entity'):
                    dct[cls].append((val[0], val[1], val[2]))
                else:   
                    inv_cnt += 1
                    inv_dct[img_path] += 1
            elif len(val) == 3 and val[0].startswith('entity') and val[1].startswith('entity'):
                if val[0] in dct['entity'] and val[1] in dct['entity']:
                    if val[0] in val[2] and val[0] in val[2]:
                        dct[cls].append((val[0], val[1], val[2]))
                    else:
                        inv_cnt += 1
                        inv_dct[img_path] += 1
                else:
                    inv_cnt += 1
                    inv_dct[img_path] += 1
            else:
                e1 = val[0]
                e2 = val[1]
                if not e1.startswith('entity'):
                    e1 = 'entity {}'.format(e1)
                if not e2.startswith('entity'):
                    e2 = 'entity {}'.format(e2)
                if e1 in dct['entity'] and e2 in dct['entity']:
                    if e1 in val[2] and e2 in val[2]:
                        dct[cls].append((e1, e2, val[2]))
                    else:
                        inv_cnt += 1
                        inv_dct[img_path] += 1
                else:
                    inv_cnt += 1                                
                    inv_dct[img_path] += 1

        elif cls == 'global':
            dct['global'].append(val[0])
        else:
            assert cls == 'location'
            val[1] = str2loc(val[1])
            if val[1] is None:
                inv_cnt += 1
                inv_dct[img_path] += 1
            else:
                if val[0] in dct['entity']:
                    dct['location'][val[0]] = val[1]
                elif not val[0].startswith('entity') and 'entity {}'.format(val[0]) in dct['entity']:
                    dct['location']['entity {}'.format(val[0])] = val[1]
                else:
                    inv_cnt += 1
                    inv_dct[img_path] += 1
            if idx == len(str_list)+1:
                str_list.append(content)
            else:
                str_list.append(content.replace(str(idx), str(len(str_list)+1))+'\n')

    return dct, inv_cnt, tot_cnt, inv_dct, str_list


def get_content_dct(filename, key):
    with open(filename) as f:
        cand_list = json.load(f)

    tot_inv_dct = {}
    tot_inv_cnt = 0
    tot_tot_cnt = 0
    cand_dct = {}
    strs_dct = {}
    for dct in cand_list:
        image_path = dct['image_path']
        image_path = '/'.join(image_path.split('/')[-3:])
        content_str = dct[key]
        res_dct, inv_cnt, tot_cnt, inv_dct, strs = parse_content(content_str, image_path, mode=key)
        cand_dct[image_path] = res_dct
        tot_inv_cnt += inv_cnt
        tot_tot_cnt += tot_cnt
        if inv_dct[image_path]:
            tot_inv_dct |= inv_dct
        strs_dct[image_path] = strs

    return cand_dct, strs_dct


def get_synonyms(word, pos=None):
    if pos is not None:
        synsets = wn.synsets(word, pos)
    else:
        synsets = wn.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def compute_synonyms_score(word1, word2, pos=None):
    if word1 in word2 or word2 in word1:
        return 1
    elif len(word1.split()) > 0 or len(word2.split() > 0):
        pass
    synonyms1 = set({})
    for w in word1.split():
        synonyms1 |= get_synonyms(w, pos)
    synonyms2 = set({})
    for w in word2.split():
        synonyms2 |= get_synonyms(w, pos)
    iou = len(synonyms1.intersection(synonyms2)) / (len(synonyms1.union(synonyms2)) + 1e-6)
    return iou


'''
    Four rules for calculating how two entities are matched, sorted by priority:
        1. the two entity names are the same
        2. the two entity names are synonyms
        3. the two entity names have similar sentence embeddings
        4. the two entity have close locations (used when at least one of above three conditions is met)
'''
def eval_entity(gt_items, cand_items, gt_locs, cand_locs):
    # 
    # calculate sentence embeddings and similarities for soft matching
    gval_list = [gval for gidx, gval in gt_items.items()]
    cval_list = [cval for cidx, cval in cand_items.items()]
    gfeats, cfeats = encode_phrases(text_encoder, gval_list, cval_list, batch_size=1)
    sent_sim = gfeats.dot(cfeats.T)
    # 
    # calculate one-to-one mapping
    gval_list = [(gidx, gval) for gidx, gval in gt_items.items()]
    cval_list = [(cidx, cval) for cidx, cval in cand_items.items()]
    cost_mat = np.zeros((len(gval_list), len(cval_list)))
    loc_flag = np.zeros((len(gval_list), len(cval_list)))
    for ii, (gidx, gval) in enumerate(gval_list):
        gloc = gt_locs[gidx] if gidx in gt_locs else None
        for jj, (cidx, cval) in enumerate(cval_list):
            cloc = cand_locs[cidx] if cidx in cand_locs else None
            if gval == cval:                                                    # entity name
                cost_mat[ii, jj] = -1000
            else:
                syn_score = compute_synonyms_score(gval, cval, wn.NOUN)
                if syn_score > 0:                                               # entity name synonyms
                    cost_mat[ii, jj] = -100 * syn_score
                elif sent_sim[ii, jj] >= ethr:                                  # similarity between sentence embeddings larger than a threshold
                    cost_mat[ii, jj] = -10 * sent_sim[ii, jj]
                else:                                                           # not matched
                    cost_mat[ii, jj] = 1e6
            if cost_mat[ii, jj] < 0 and gloc is not None and cloc is not None:      # location
                biou = box_iou(torch.tensor([gloc]), torch.tensor([cloc]))[0][0].item()
                if biou >= lthr:
                    cost_mat[ii, jj] += -1 * biou
                    loc_flag[ii, jj] = 1
    gt_idx, cand_idx = linear_sum_assignment(cost_mat)
    # 
    # calculate entity score based on one-to-one mapping
    erec = 0
    eprec = 0
    lrec = 0
    lprec = 0
    emap = {}
    for ii, jj in zip(gt_idx, cand_idx):
        if cost_mat[ii, jj] < 0:
            erec += 1
            eprec += 1
            if loc_flag[ii, jj] > 0.5:
                lrec += 1
                lprec += 1
            emap[gval_list[ii][0]] = cval_list[jj][0]
    erec /= len(gt_items)
    if len(cand_items) == 0:
        eprec = 0
    else:
        eprec /= len(cand_items)
    if len(gt_locs) > 0:
        lrec /= len(gt_locs)
    else:
        lrec = 0.0
    if len(cand_locs) > 0:
        lprec /= len(cand_locs)

    if erec+eprec == 0.0:
        ef1 = 0.0
    else:
        ef1 = 2*erec*eprec/(erec+eprec)
    if lrec+lprec == 0.0:
        lf1 = 0.0
    else:
        lf1 = 2*lrec*lprec/(lrec+lprec)

    return erec, eprec, ef1, lrec, lprec, lf1, emap

def eval_attr(gt_attrs, cand_attrs, gt_ans_attrs, cand_ans_attrs):
    if len(gt_attrs) == 0: return 0, 0, 0
    arec = 0.0
    for ii, (aidx, aattr) in enumerate(gt_ans_attrs):
        ans1 = aattr.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = aattr.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            arec += 0.5
        if ans2 == 'no':
            arec += 0.5
        if ii >= len(gt_attrs): break
    aprec = 0.0
    for ii, (aidx, aattr) in enumerate(cand_ans_attrs):
        ans1 = aattr.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = aattr.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            aprec += 0.5
        if ans2 == 'no':
            aprec += 0.5
        if ii >= len(cand_attrs): break
    arec /= len(gt_attrs)
    if len(cand_ans_attrs) == 0 or len(cand_attrs) == 0:
        aprec = 0.0
    else:
        aprec /= len(cand_ans_attrs)
    if arec+aprec == 0.0:
        af1 = 0.0
    else:
        af1 = 2*arec*aprec/(arec+aprec)
    return arec, aprec, af1

def eval_relation(gt_rels, cand_rels, gt_ans_rels, cand_ans_rels):
    if len(gt_rels) == 0: return 0, 0, 0
    rrec = 0.0
    for ii, (aidx1, aidx2, arel) in enumerate(gt_ans_rels):
        ans1 = arel.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = arel.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            rrec += 0.5
        if ans2 == 'no':
            rrec += 0.5
        if ii >= len(gt_rels): break
    rprec = 0.0
    for ii, (aidx1, aidx2, arel) in enumerate(cand_ans_rels):
        ans1 = arel.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = arel.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            rprec += 0.5
        if ans2 == 'no':
            rprec += 0.5
        if ii >= len(cand_rels): break
    rrec /= len(gt_rels)
    if len(cand_ans_rels) == 0 or len(cand_rels) == 0:
        rprec = 0.0
    else:
        rprec /= len(cand_ans_rels)
    if rrec+rprec == 0.0:
        rf1 = 0.0
    else:
        rf1 = 2*rrec*rprec/(rrec+rprec)
    return rrec, rprec, rf1

def eval_global(gt_glos, cand_glos, gt_ans_glos, cand_ans_glos):
    if len(gt_glos) == 0: return 0, 0, 0
    grec = 0
    for ii, aglo in enumerate(gt_ans_glos):
        ans1 = aglo.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = aglo.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            grec += 0.5
        if ans2 == 'no':
            grec += 0.5
        if ii >= len(gt_glos): break
    gprec = 0
    for ii, aglo in enumerate(cand_ans_glos):
        ans1 = aglo.split(',')[0].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        ans2 = aglo.split(',')[1].lstrip('\'\" ,').rstrip('\'\" ,').lower()
        if ans1 == 'yes':
            gprec += 0.5
        if ans2 == 'no':
            gprec += 0.5
        if ii >= len(cand_glos): break
    grec /= len(gt_glos)
    if len(cand_ans_glos) == 0 or len(cand_glos) == 0:
        gprec = 0.0
    else:
        gprec /= len(cand_ans_glos)
    if grec+gprec == 0.0:
        gf1 = 0.0
    else:
        gf1 = 2*grec*gprec/(grec+gprec)
    return grec, gprec, gf1


def eval_all(gt_dcts, cand_dcts, gt_ans_dcts, cand_ans_dcts):
    # tagging
    erec_list = []
    eprec_list = []
    ef1_list = []
    # location
    lrec_list = []
    lprec_list = []
    lf1_list = []
    # attribute
    arec_list = []
    aprec_list = []
    af1_list = []
    # relation
    rrec_list = []
    rprec_list = []
    rf1_list = []
    # global
    grec_list = []
    gprec_list = []
    gf1_list = []

    for img, gt_dct in tqdm.tqdm(gt_dcts.items()):
        cand_dct = cand_dcts[img]
        gt_ans_dct = gt_ans_dcts[img]
        cand_ans_dct = cand_ans_dcts[img]
        erec, eprec, ef1, lrec, lprec, lf1, _ = eval_entity(gt_dct['entity'], cand_dct['entity'], gt_dct['location'], cand_dct['location'])

        # evaluation based on answers
        arec, aprec, af1 = eval_attr(gt_dct['attribute'], cand_dct['attribute'], gt_ans_dct['attribute'], cand_ans_dct['attribute'])
        rrec, rprec, rf1 = eval_relation(gt_dct['relation'], cand_dct['relation'], gt_ans_dct['relation'], cand_ans_dct['relation'])
        grec, gprec, gf1 = eval_global(gt_dct['global'], cand_dct['global'], gt_ans_dct['global'], cand_ans_dct['global'])

        erec_list.append(erec)
        eprec_list.append(eprec)
        ef1_list.append(ef1)
        lrec_list.append(lrec)
        lprec_list.append(lprec)
        lf1_list.append(lf1)
        arec_list.append(arec)
        aprec_list.append(aprec)
        af1_list.append(af1)
        rrec_list.append(rrec)
        rprec_list.append(rprec)
        rf1_list.append(rf1)
        grec_list.append(grec)
        gprec_list.append(gprec)
        gf1_list.append(gf1)

    avg_erec = sum(erec_list) / len(erec_list)
    avg_eprec = sum(eprec_list) / len(eprec_list)
    avg_ef1 = sum(ef1_list) / len(ef1_list)
    avg_lrec = sum(lrec_list) / len(lrec_list)
    avg_lprec = sum(lprec_list) / len(lprec_list)
    avg_lf1 = sum(lf1_list) / len(lf1_list)
    avg_arec = sum(arec_list) / len(arec_list)
    avg_aprec = sum(aprec_list) / len(aprec_list)
    avg_af1 = sum(af1_list) / len(af1_list)
    avg_rrec = sum(rrec_list) / len(rrec_list)
    avg_rprec = sum(rprec_list) / len(rprec_list)
    avg_rf1 = sum(rf1_list) / len(rf1_list)
    avg_grec = sum(grec_list) / len(grec_list)
    avg_gprec = sum(gprec_list) / len(gprec_list)
    avg_gf1 = sum(gf1_list) / len(gf1_list)
    print('\tTagging F1-Score:', avg_ef1)
    print('\tLocalization F1-Score:', avg_lf1)
    print('\tAttribute F1-Score:', avg_af1)
    print('\tRelation F1-Score:', avg_rf1)
    print('\tGlobal F1-Score:', avg_gf1)
    
    pancapscore = avg_ef1 + avg_lf1 + avg_af1 + avg_rf1 + 0.1*avg_gf1
    print('Overall PancapScore:', pancapscore)

if __name__ == '__main__':
    print('Evaluating all metrics...')

    gt_dct, _ = get_content_dct(gt_file, 'questions')
    cand_dct, _ = get_content_dct(cand_file, 'all_content')
    gt_ans_dct, _ = get_content_dct(gt4pred_answer_file, 'answers')
    cand_ans_dct, _ = get_content_dct(pred4gt_answer_file, 'answers')

    eval_all(gt_dct, cand_dct, gt_ans_dct, cand_ans_dct)


