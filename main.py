import pandas as pd
import numpy as np
import re
import string
import os
import json
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

comp_index = pd.read_csv(os.path.expanduser("~/Downloads/kospi200_info_v3_Ahmet.csv"))
df_data = pd.read_parquet(os.path.expanduser("~/Downloads/NewsData_LG-Electronics.parquet"), engine="auto")

comp_index.eng_name_clean = comp_index.eng_name_clean.astype(str)
clean_names = comp_index.eng_name_clean.tolist()

# forming a reference dictionary to extract id and lookup name later

name_id_dict = {}
for i in range(len(clean_names)):
    name_id_dict[comp_index.at[i, "eng_name_clean"].upper()] = (comp_index.at[i, "corp_code"], comp_index.at[i, "eng_name"].upper(), comp_index.at[i, "eng_name_clean"].upper())

# preprocessing in progress


def extract4sentence(text):
    eng = re.compile('[\n]')
    result = eng.sub(' ', text)
    result = ' '.join(result.split())
    return result


def simp_preprop(df):
    print('Removing missing values...')
    df = df[~df['summary'].isna()].reset_index()

    print('Preprocessing...')
    contents = df.title + '.\n' + df.summary
    df['contents'] = contents.apply(lambda x: extract4sentence(x))

    # sentseg = spacy.load("en_core_web_md")
    # df['contents'] = df['contents'].apply(lambda x : '. '.join([x.text for x in sentseg(x).sents]))

    return df


df_trial = simp_preprop(df_data)

nlp = spacy.load("en_core_web_trf", exclude=["tagger", "lemmatizer", "morphologizer", "textcat"])
text = df_trial.contents[7]
document = nlp(text)


def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


def lowercase_translator(my_string):
        return my_string.lower().translate(str.maketrans('', '', string.punctuation))


def det_comp_list(doc):
        detected_list = []
        for ent in doc.ents:
            if ent._.is_company == "COMP":
                detected_list.append(ent)
        return detected_list


def longest_company_name(a_list, doc):
    if len(a_list) > 0:
        long_comp = a_list[0]
        for a in a_list:
            if len(a.text) > len(long_comp.text):
                long_comp = a
    else:
        long_comp = doc[0:0]
    return long_comp


def NLP_results(doc, index_dict):

    names = [key for key in index_dict]

    # building up patterns and matcher

    patterns = list(nlp.pipe(names))
    final_pattern = []
    matcher = Matcher(nlp.vocab)
    for i in range(len(patterns)):
        another_list = []
        for token in patterns[i]:
            another_list.append({"LOWER": token.text.lower()})
        final_pattern.append(another_list)
        final_pattern.append([{"LOWER": "samsung"}])
        matcher.add(names[i].upper(), final_pattern)
        final_pattern = []
    matches = matcher(doc)

    # setting extensions and deciding company attribute of recognized entities

    Span.set_extension("is_company", default=None, force=True)
    Span.set_extension("lookup_name", default=None, force=True)
    Span.set_extension("company_id", default=None, force=True)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            for match_id, start, end in matches:
                if nlp.vocab.strings[match_id] in ent.text.lower():
                    ent._.is_company = "COMP"
                elif lowercase_translator(ent.text) in lowercase_translator(nlp.vocab.strings[match_id]):
                    ent._.is_company = "COMP"
                elif lowercase_translator(nlp.vocab.strings[match_id]) in lowercase_translator(ent.text):
                    ent._.is_company = "COMP"

    a_list = det_comp_list(doc)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            for a in a_list:
                if a.text in ent.text:
                    ent._.is_company = "COMP"


    # setting extensions and deciding lookup name and ID attributes of recognized entities

    for ent in doc.ents:
        if ent.label_ == "ORG":
            for match_id, start, end in matches:
                if nlp.vocab.strings[match_id] in ent.text.lower():
                    ent._.lookup_name = index_dict[nlp.vocab.strings[match_id]][1]
                    ent._.company_id = index_dict[nlp.vocab.strings[match_id]][0]
                elif lowercase_translator(nlp.vocab.strings[match_id]) in lowercase_translator(ent.text):
                    ent._.lookup_name = index_dict[nlp.vocab.strings[match_id]][1]
                    ent._.company_id = index_dict[nlp.vocab.strings[match_id]][0]

    # linking non-pronoun references to main company entity info (used a simple logic: longest is the best)

    while len(a_list) > 1:
        long_comp = longest_company_name(a_list, doc)
        for elm in a_list:
            if elm.text in long_comp.text and elm is not long_comp:
                elm._.lookup_name = long_comp._.lookup_name
                elm._.company_id = long_comp._.company_id
                a_list.remove(elm)
        a_list.remove(long_comp)

    # preparing objects and values for final dictionary buildup

    sent_num = len([sent for sent in doc.sents])

    sent_i_dict = {}
    for sent_index, sent in enumerate(doc.sents):
        sent_i_dict[sent] = sent_index

    sent_with_ent = set()
    for ent in doc.ents:
        sent_with_ent.add(sent_i_dict[ent.sent])
    sent_with_ent_num = len(sent_with_ent)

    sent_with_comp = set()
    for ent in doc.ents:
        if ent._.is_company == "COMP":
            sent_with_comp.add(sent_i_dict[ent.sent])
    sent_with_comp_num = len(sent_with_comp)

    detected_companies_set = set()
    for ent in doc.ents:
        if ent._.is_company == "COMP":
            detected_companies_set.add(ent.text)
            detected_companies_set.add(ent._.lookup_name)

    freq_list = list(doc.ents[:])
    freq_dict = {}
    for ent in freq_list:
        if ent._.lookup_name is not None:
            if ent._.lookup_name in freq_dict.keys():
                freq_dict[ent._.lookup_name][0] += 1
                freq_dict[ent._.lookup_name][1].append(sent_i_dict[ent.sent])
            else:
                freq_dict[ent._.lookup_name] = [1, [sent_i_dict[ent.sent]]]
        else:
            if ent.text in freq_dict.keys():
                freq_dict[ent.text][0] += 1
                freq_dict[ent.text][1].append(sent_i_dict[ent.sent])
            else:
                freq_dict[ent.text] = [1, [sent_i_dict[ent.sent]]]
    for key in freq_dict.keys():
        freq_dict[key].append(round(freq_dict[key][0]/sent_num, 2))
        freq_dict[key].append(round(freq_dict[key][0]/sent_with_ent_num, 2))
        if key in detected_companies_set:
            freq_dict[key].append(round(freq_dict[key][0]/sent_with_comp_num, 2))

    # list_of_dict_per_ent --> lodpe
    # dict_per_ent (explanatory naming) = {Entity Name, Entity Label, Company Recognizer,
    #                                      Lookup Name, Company ID, Frequency,
    #                                      Parent Sentence Indices,
    #                                      Ratio with respect to number of all sentences,
    #                                      Ratio with respect to number of sentences including entities,
    #                                      Ratio with respect to sentences containing company entities}
    # dict_per_ent (real naming) = {name, label, comp_or_not, lookup_name,
    #                               comp_id, freq, parent_sent_i, ratio_wrt_all_sent,
    #                               ratio_wrt_ent_sent, ratio_wrt_comp_sent}

    lodpe = []
    name_list = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "PRODUCT", "GPE"]:
            if ent.text not in name_list and ent._.lookup_name not in name_list:
                dict_per_ent = {"name": ent.text, "label": ent.label_, "comp_or_not": ent._.is_company,
                                "lookup_name": ent._.lookup_name, "comp_id": ent._.company_id}
                if ent._.lookup_name in freq_dict.keys():
                    dict_per_ent["name"] = ent._.lookup_name
                    dict_per_ent["freq"] = freq_dict[ent._.lookup_name][0]
                    dict_per_ent["parent_sent_i"] = freq_dict[ent._.lookup_name][1]
                    dict_per_ent["ratio_wrt_all_sent"] = freq_dict[ent._.lookup_name][2]
                    dict_per_ent["ratio_wrt_ent_sent"] = freq_dict[ent._.lookup_name][3]
                    dict_per_ent["ratio_wrt_comp_sent"] = freq_dict[ent._.lookup_name][4]
                elif ent._.is_company == "COMP":
                    dict_per_ent["freq"] = freq_dict[ent.text][0]
                    dict_per_ent["parent_sent_i"] = freq_dict[ent.text][1]
                    dict_per_ent["ratio_wrt_all_sent"] = freq_dict[ent.text][2]
                    dict_per_ent["ratio_wrt_ent_sent"] = freq_dict[ent.text][3]
                    dict_per_ent["ratio_wrt_comp_sent"] = freq_dict[ent.text][4]
                else:
                    dict_per_ent["freq"] = freq_dict[ent.text][0]
                    dict_per_ent["parent_sent_i"] = freq_dict[ent.text][1]
                    dict_per_ent["ratio_wrt_all_sent"] = freq_dict[ent.text][2]
                    dict_per_ent["ratio_wrt_ent_sent"] = freq_dict[ent.text][3]
                lodpe.append(dict_per_ent)
                name_list.append(ent.text)
                if ent._.lookup_name is not None:
                    name_list.append(ent._.lookup_name)

    # lodpe = json.dumps(lodpe, default = myconverter)
    return(lodpe)

print(NLP_results(document, name_id_dict))
