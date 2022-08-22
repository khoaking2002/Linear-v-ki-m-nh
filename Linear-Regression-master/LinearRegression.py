from utils import *

def gradient(X,w,y):
  n = X.shape[0]
  return 1/n*np.dot(X.T,np.dot(X.T,w)-y)

def cost(X,w,y):
  n = X.shape[0]
  return 0.5/n*np.linalg.norm(y - np.dot(X,w),2)**2

def Gradient_descent(X,y,w_init,lr,iter):
  w = [w_init]
  for it in range(iter):
    w_new = w[-1] - lr*gradient(X,w[-1],y)
    if (np.linalg.norm(gradient(X,w_new,y))/len(w_new) < 1e-3):
      break
    w.append(w_new)
  return (w,it)

class LinearRegression:
  def __init__(self,graphfile):
    self. w = []
    self.y0 = 0
    self.X =[]
    self.y = []
    self.x0 = np.linspace(0,800,2)
    self.graph_file = graphfile
  def fit(self,X,y):
    self.X = X
    self.y = y
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one,X),axis = 1)
    a = np.dot(Xbar.T,Xbar)
    b = np.dot(Xbar.T,y)
    w = np.dot(np.linalg.pinv(a),b)
    self.w = w 
    print("Weight of the model",self.w)
    print("Finish update weight")
  def get_weight(self):
    print("weight of the model",self.w)
    return (self.w)
  def plot_graph(self):
    if (len(self.w)>2):
      print("Does not support at the momment, try later")
      return False
    if self.X == [] or self.y == []:
      print("fail to plot, please add data")
      return False
    else:
      # plt.plot(self.X.T, self.y.T, 'ro') 
      plt.scatter(self.X, self.y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
      plt.plot(self.x0, self.y0)             
      plt.xlabel('target temperature')
      plt.ylabel('prcp')
      # plt.show()
      plt.savefig(self.graph_file)
      return True
  def predict_one_value(self,x):
    res = 0
    for i in range(1,len(self.w)):
      res += self.w[i][0]*x[i-1]
    res += self.w[0][0]
    return np.round(res,5)
  def predict(self,X):
    res = []
    for x in X:
      res.append(self.predict_one_value(x))
    return res
  def mse(self,y_predict,y_true):
    res = 0
    m = len(y_true)
    for i in range(len(y_true)):
      res+= (y_true[i] - y_predict[i])**2
    return float(res/m)
  def mae(self,y_predict,y_true):
    res = 0
    m = len(y_true)
    for i in range(len(y_true)):
      res+= abs(y_true[i] - y_predict[i])
    return float(res/m)