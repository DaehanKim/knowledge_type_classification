# preprocess data
import pandas as pd 
import numpy as np


def extract_data_list():
	relevant_cols = ['정의','과정','성질','예','흥미유발']
	label_map = dict(zip(relevant_cols, range(5)))

	src1 = pd.read_excel('중1.xlsx', sheet_name='Merge')
	src2 = pd.read_excel('중2.xlsx', sheet_name='Merge')
	src3 = pd.read_excel('중3.xlsx', sheet_name='Sheet1 (2)')

	src_list = [src1, src2, src3]
	data_list = []

	for src in src_list:
		for col in relevant_cols:
			data_list += [(item,label_map[col]) for item in src[col][~src[col].isnull()]]

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