from pororo import Pororo
import torch
import pandas as pd
import string
import os


ner_ko = Pororo(task='ner', lang='ko')
ner_en = Pororo(task='ner', lang='en')
txt_ko = '페어랩스는 손종진 대표가 설립한 스타트업 회사로 2021년 7월에 데이터 서비스를 제공하고 있으며 KISTEP, KISTI 등에게 팩트셋의 데이터 분석서비스를 제공한다.'
txt_en = '''APG Asset Management, Europe's largest pension investor, has started pressuring Korean conglomerates to respond more proactively to climate change. Operating about 850 trillion won ($709 billion) worth of assets and investing for over 4.7 million Dutch pension participants, APG sent letters recently to 10 Korean companies ― all affiliates of major Korean conglomerates ― urging them to make greater efforts to cut carbon emissions. APG sent the letters to CEOs and chairs of 10 Korean companies ― Samsung Electronics, SK hynix, SK Inc., LG Chem, LG Display, Lotte Chemical, Hyundai Steel, POSCO Chemical, LG Uplus and SK Telecom. APG has around 3 trillion won invested in these 10 Korean companies. In its letter to Samsung Electronics, APG stated that the firm's carbon emissions-to-revenue ratio stood at 8.7 percent as of 2020, which is a much higher level compared to its global IT tech peers. Apple's emissions-to-revenue ratio is around 0.3 percent. If Samsung Electronics fails to reduce its current level of carbon emissions, its valuation will be at risk of decreasing due to an increase in emissions costs, it said. Regarding SK hynix, APG pointed out that the firm's emissions-to-revenue ratio is about three times higher than Samsung Electronics'. As for SK Inc., the Dutch pension investor said, "Despite its plan to achieve net zero by 2050, the company has no specific timeline or plans." APG asked the firm to share more detailed plans with investors. Companies like LG Chem, POSCO Chemical and Lotte Chemical were also urged to participate in the green initiative, given that their carbon emissions compared to their revenues are at a higher level than other global chemical companies. As it is rare for overseas pension investors to disclose detailed information on their invested companies, the move is seen as APG's strong drive to pressure the companies that fail to make meaningful changes in their carbon reduction policies. An APG official said that considering the 10 companies' status within global supply chains, as well as Korea's economic size, they haven't responded fully and quickly to climate change risks. APG also plans to closely cooperate with other institutional shareholders of the 10 companies to maximize shareholder engagement, sending warning messages that corporate activities should not be pursued in a way that harms the environment and humanity.'''
res_ko = ner_ko.predict(txt_ko)
res_en = ner_en.predict(txt_en)


# NER Detection (ko)
orgs_ko = [item for item in res_ko if 'ORGANIZATION' in item[1]] 


# NER Detection (en)
# orgs_prep is a list of recognized entities. These entities are undergone preprocessing including replacing 's suffix and punctuations which hinder company tagging process.
orgs = [item for item in res_en if 'ORG' in item[1]]
orgs_prep = []
for item in orgs:
  if item[0].endswith("'s"):
    item = (item[0].replace("'s", ""), 'ORG')
  item = (item[0].replace(',', '').replace('.', '').strip(), 'ORG')
  orgs_prep.append(item)


comp_index = pd.read_csv(os.path.expanduser('~/Downloads/kospi200_info_v4_company_table.csv'))
# name_id_dict is a dictionary having clean company names as its keys and (corporation code, english name, clean english name) tuples as values mapped by keys.
clean_names = comp_index.eng_name_clean.tolist()
name_id_dict = {}
for i in range(len(clean_names)):
  name_id_dict[comp_index.at[i, "eng_name_clean"]] = (comp_index.at[i, "corp_code"], comp_index.at[i, "eng_name"].upper(), comp_index.at[i, "eng_name_clean"].upper())


company_table = comp_index[['corp_code', 'eng_name', 'eng_name_clean', 'stock_code']]


# The function below gets an input text and returns it without punctuations and in lowercase form.
def lowercase_translator(my_string):
  return my_string.lower().translate(str.maketrans('', '', string.punctuation))


# Process below matches company entities with their values in index and results in a list of company index series.
super_x = []
for item in orgs_prep:
  entity_name = item[0]
  entity_name_clean = lowercase_translator(entity_name)
  matching = company_table[company_table.eng_name_clean.str.lower() == entity_name_clean]
  super_x.append(matching)


# List of series is converted into a data frame for the sake of further applications. 
super_x = pd.concat(super_x)


from nltk import sent_tokenize
sents = sent_tokenize(txt_en)


# parent_sent_i --> key: recognized entity name, value: list of sentences containing that entity
parent_sent_i = {}
for item in orgs_prep:
  entity_name = item[0]
  idx = []
  for i in range(len(sents)):
    if entity_name in sents[i]:
      idx.append(i)
  parent_sent_i[entity_name] = idx


# clone of parent_sent_i 
parent_sent_i_v2 = {}
for key in list(parent_sent_i):
  parent_sent_i_v2[key] = parent_sent_i[key]


detected_comp_list = list(super_x.eng_name_clean.astype(str))
final_dict = {}
list_of_orgs = list(parent_sent_i)
# in first part (for loop), clean names of matched companies are added into final_dict as key
for i in detected_comp_list:
  final_dict[i] = []
# in second part (nested for loop), the keyword "added" is appended into sentence index list of a key (value of the key) if that key is in the matched company names
# lowercase_translator checks are performed to get rid of case sensitivities and possible problems due to non-alphanumeric characters
# company entities are keyed into their respective sentence index lists after these processes
for key in list(final_dict):
  for key2 in list_of_orgs:
    if key in key2 or key2 in key:
      final_dict[key].extend(parent_sent_i[key2])
      parent_sent_i_v2[key2].append("added")
    elif lowercase_translator(key2) in lowercase_translator(key):
      final_dict[key].extend(parent_sent_i[key2])
      parent_sent_i_v2[key2].append("added")
    elif lowercase_translator(key) in lowercase_translator(key2):
      final_dict[key].extend(parent_sent_i[key2])
      parent_sent_i_v2[key2].append("added")
# third part (for loop) is adding all non-company entities into final_dict as they require no corrections
for key in list_of_orgs:
  if "added" not in parent_sent_i_v2[key]:
    final_dict[key] = parent_sent_i[key]
# lastly, deduplication is performed in values and control keywords are removed
for key in list(final_dict):
  xyz = final_dict[key]
  xyz = list(set(xyz))
  final_dict[key] = xyz
  if "added" in final_dict[key]:
    final_dict[key].remove("added")


# sets of sentences containing entities and companies, respectively
sent_with_ent_i = set()
sent_with_comp_i = set()
for key in list(final_dict):
  if key in list(name_id_dict):
    for i in final_dict[key]:
      sent_with_ent_i.add(i)
      sent_with_comp_i.add(i)
  else:
    for i in final_dict[key]:
      sent_with_ent_i.add(i)


# numbers of sentences containing entities and companies, respectively
sent_num = len(sents)
sent_with_ent_num = len(sent_with_ent_i)
sent_with_comp_num = len(sent_with_comp_i)


# clean super_x table
super_x = super_x.drop_duplicates().reset_index(drop=True)


# list of entity dictionaries
lodpe = []
for key in list(final_dict):
  d = {}
  d["name"] = key
  d["label"] = "ORG"
  d["freq"] = len(final_dict[key])
  d["parent_sent_i"] = final_dict[key]
  d["ratio_wrt_all_sent"] = round(d["freq"]/sent_num, 2)
  d["ratio_wrt_ent_sent"] = round(d["freq"]/sent_with_ent_num, 2)
  if key in list(name_id_dict):
    d["comp_or_not"] = "COMP"
    d["lookup_name"] = name_id_dict[key][1]
    d["comp_id"] = name_id_dict[key][0]
    d["ratio_wrt_comp_sent"] = round(d["freq"]/sent_with_comp_num, 2)
  else:
    d["comp_or_not"] = None
    d["lookup_name"] = None
    d["comp_id"] = None
    d["ratio_wrt_comp_sent"] = None
  lodpe.append(d)
for k in lodpe:
  print(k)