from utils import *
from LinearRegression import LinearRegression

# from sklearn.metrics import mean_absolute_error
linear = LinearRegression(graphfile)
X,y = create_data(csv_file,X_name,y_name)
print(X.shape)

x_train,y_train,x_test,y_test = train_test_split(X,y,0.2)
print("Starting update weight ...")
linear.fit(X,y)

print("plot data distribution in 3D")
plot_3d_graph(X,y)

print("plot data distribution in 2D")
plot_2d_graph(X,y)

print("Weight after update:")
linear.get_weight()

sample = readfile(inputfile)
print(sample)

print("Predicting phase:")
out = []
for i in range(len(sample)):
	print("Test {}:".format(i),linear.predict_one_value(sample[i]))
	out.append(linear.predict_one_value(sample[i]))
writefile(outfile,out)

y_predict = linear.predict(x_test)

print("comparing with the true label")
compare(y_test,y_predict)

print("Graph for the linear on test set")
plot_line_2d(y_test,y_predict)

print("Testing model:")

mae = linear.mae(y_predict,y_test)
mse = linear.mse(y_predict,y_test)
print("Mae: ",mae)
print("MSE: ",mse)

f = open(evalfile,"w")
f.write("Mae score: " + str(mae) +"\n"
		+"Mse score: " + str(mse)+"\n")

