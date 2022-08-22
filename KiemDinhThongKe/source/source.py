import scipy.stats as st
import csv
import math
import matplotlib.pyplot as plt
#Tinh gia tri trung binh
def means(Lst):
	s=0
	for i in Lst:
		s=s+i
	return s/(len(Lst))
#Tinh do lech chuan mau
def S(Lst):
	mean_Lst=means(Lst)
	s_2=0
	for i in Lst:
		s_2=s_2+(i-mean_Lst)*(i-mean_Lst)
	return math.sqrt(s_2/(len(Lst)-1))
#Tinh gia tri thong ke kiem dinh
def z(Lst1,Lst2):
	delta_means=means(Lst1)-means(Lst2)
	delta_S=(S(Lst1)*S(Lst1))/len(Lst1)+(S(Lst2)*S(Lst2))/len(Lst2)
	return delta_means/math.sqrt(delta_S)
#Kiem dinh voi hai danh sach va alpha
def KiemDinh(Lst1,Lst2,alpha):
	kdtk=z(Lst1,Lst2)
	z_alpha=st.norm.ppf(1-alpha)
	print('z_1-alpha is: ',z_alpha,'\n','z-kiem dinh is: ', kdtk,'\n')
	if(kdtk>z_alpha):
		print('Luong mua toi da trung binh cua khu vuc trong nam cua range station lon hon caliente')
	else:
		print('Luong mua toi da trung binh cua khu vuc trong nam range station khong lon hon caliente')
#Doc file CSV
def read_csv(link):
    with open(link) as f:
    	Lst=[]
    	reader=csv.reader(f)
    	for row in reader:
    		if(row[6]!='' and row[6]!='EMXP'):
    			Lst.append(float(row[6]))
    return Lst
#Kiem chung voi nhieu gia tri alpha 
Lst_1=read_csv('../data/range_station.csv')
Lst_2=read_csv('../data/caliente.csv')
alpha_list=[0.2,0.15,0.1,0.05,0.02,0.01,0.001,0.0001,0.00001]
for i in alpha_list:
	print('Muc y nghia', i)
	KiemDinh(Lst_1,Lst_2,i)
#Truc quan hoa du lieu
plt.plot(Lst_1,color='r',label='range_station')
plt.plot(Lst_2,color='b',label='caliente')
plt.ylabel("Tens of milimeter")
plt.title("EXMP in range_station and caliente")
plt.legend()
plt.show()