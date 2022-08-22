import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

X_name = ["HLY-TEMP-NORMAL","HLY-PRES-NORMAL"]
y_name = ["HLY-DEWP-NORMAL"]
csv_file = "NORMAL_HLY_sample_csv.csv"
inputfile ="./asset/test.txt"
outfile = "./asset/out.txt"
graphfile = "./asset/graph_out.png"
evalfile = "./asset/evaluation.txt"
graph2d = "./asset/distribution_2d.png"
graph3d = "./asset/distribution_3d.png"
comparegraph = "./asset/comparegraph.png"
def create_data(csv_file,X_name,y_name):
	X = []
	y = []
	df = pd.read_csv(csv_file)
	for name in X_name:
		tmp = []
		for i in range(len(df)):
			tmp.append(df[name][i])
		X.append(tmp)
	for name in y_name:
		tmp = []
		for i in range(len(df)):
			tmp.append(df[name][i])
		y.append(tmp)
	return np.array(X).T,np.array(y).T

def readfile(inputfile):
	list_of_lists = []
	with open(inputfile) as f:
		for line in f:
			inner_list = [int(elt.strip()) for elt in line.split()]
			list_of_lists.append(inner_list)
	return list_of_lists

def writefile(outfile, output):
	out = open(outfile,"w") 
	for i in range(len(output)):
		out.write("Prediction of test {}: ".format(i+1)+ str(output[i]) +"\n")
	print("Finished")

def train_test_split(X,y, ratio = 0.2):
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	n_data = math.floor(len(X)*ratio)
	if (len(X)!=len(y)):
		print("No same shape of X and Y")
		return None
	for i in range(n_data):
		train_x.append(X[i])
		train_y.append(y[i])
	for i in range(n_data,len(X)):
		test_x.append(X[i])
		test_y.append(y[i])
	return np.array(train_x), np.array(train_y),np.array(test_x),np.array(test_y)

def plot_2d_graph(X,y):
	x1 = []
	x2 = []
	for item in X:
		x1.append(item[0])
		x2.append(item[1])
	fig, ax = plt.subplots()
	scat = ax.scatter(x1,x2, c = y, s=200)
	fig.colorbar(scat)
	plt.savefig(graph2d)

def plot_3d_graph(X,y):
	x1 = []
	x2 = []
	for item in X:
		x1.append(item[0])
		x2.append(item[1])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x1, x2, y)
	plt.savefig(graph3d)

def plot_line_2d(y_test,y_pred):
	graph  = sns.regplot(x=y_test, y=y_pred, ci=None, color="b")
	graph.get_figure()
	plt.savefig(graphfile)

def compare(y_true,y_pred):
	ax1 = sns.distplot(y_true, hist=False, color="r", label="Actual Value")
	sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)
	plt.savefig(comparegraph)