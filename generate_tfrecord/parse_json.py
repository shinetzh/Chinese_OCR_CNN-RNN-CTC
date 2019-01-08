import json
import os

duan ="--------------------------"


def parse_dict(dic):
	if not isinstance(dic,dict):
		return []
	key_list = dic.keys()
	if len(key_list)>10:
		return [key_list[1]]
	return key_list


		
def parserJsonFile(jsonData):
	if not jsonData:
		return
	rootlist = parse_dict(jsonData)
	if not rootlist:
		return
	print(rootlist)
	for rootkey in rootlist:
		print(rootkey)
		parserJsonFile(jsonData[rootkey])



json_file="synth90k.json"
print("loading...")
jsondata = json.load(open(json_file, 'r'))
# parserJsonFile(jsondata)
print("loaded")
print(jsondata["337_Alike_2024.jpg"])
print(jsondata["337_Alike_2024.jpg"]["annotations"][0]["utf8_string"])