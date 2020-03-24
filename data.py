# preprocess data
import pandas as pd 
import numpy as np
import math

def extract_data_list():
	relevant_cols = ['정의','과정','성질','예','흥미유발']
	label_map = dict(zip(relevant_cols, range(5)))

	src1 = pd.read_excel('중1.xlsx', sheet_name='Merge')
	src2 = pd.read_excel('중2.xlsx', sheet_name='Merge')
	src3 = pd.read_excel('중3.xlsx', sheet_name='Sheet1 (2)')
	src_list = [src1, src2, src3]
	nums = len(src_list) # merged excel list nums
	data_list = []
	for src in src_list:
		for col in relevant_cols:
			data_list += [(item,label_map[col]) for item in src[col][~src[col].isnull()]]

	# add "한시에 데이터를 만들자"
	sheet_names = ["상효", "은지", "두희", "대한", "영근", "상헌", "근형", "민경"]
	for name in sheet_names:
		if name == "상효" or name == "은지": # 데이터가...
			continue
		src_list.append(pd.read_excel("한시에 데이터를 만들자.xlsx", sheet_name=name, header=1, keep_default_na=False))

	for index in range(nums, len(src_list)):
		src = src_list[index]
		counts = len(src)
		flag = True # 기타나 참고의 chunk를 데이터에 안넣기 위한 flag
		for i in range(counts):
			if src["단위지식 type"][i]:
				if src["단위지식 type"][i] != "기타" and src["단위지식 type"][i] != "참고":
					data_list += [(src["train_original"][i], label_map[src["단위지식 type"][i]])]
					flag = True
				else:
					flag = False
			else:
				if flag:
					temp = list(data_list[-1])
					data_list[-1] = (temp[0]+" "+src["train_original"][i], temp[1])


	print('Relevant Columns : {}'.format(' '.join(relevant_cols)))
	print('Number of Extracted Data : {}'.format(len(data_list)))

	return data_list

def write(path, data_list):
	tsv = []
	for x,y in data_list:
		tsv.append('{}\t{}\n'.format(x.replace('\n',' '),y))
	with open(path,'wt',encoding='utf8') as f:
		f.write(''.join(tsv))

def main():

	save_path = 'txt_clf_data.tsv'

	data_list = extract_data_list()
	write(save_path, data_list)


if __name__ == '__main__':
	main()