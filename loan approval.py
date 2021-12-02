# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:42:48 2021

@author: Ayush's Workhorse
"""


import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
root = tk.Tk()
canvas1=tk.Canvas(root,width=400,height=550, bg = "Yellow")
canvas1.pack()
labelT = tk.Label(root, text='WELCOME TO THE LOAN APPROVAL SYSTEM ')
canvas1.create_window(200, 50, window=labelT)
#loanid(LP00____)
label1 = tk.Label(root, text='Loan_ID(LP00____): ')
canvas1.create_window(100, 100, window=label1,)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# Gender
label2 = tk.Label(root, text=' Gender(Male or Female): ')
canvas1.create_window(100, 130, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 130, window=entry2)

label3 = tk.Label(root, text='Married(Yes or No): ')
canvas1.create_window(100, 160, window=label3)

entry3 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 160, window=entry3)

# New_Unemployment_Rate label and input box
label4 = tk.Label(root, text=' Dependents(0,1,2,3+): ')
canvas1.create_window(100, 190, window=label4)

entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 190, window=entry4)

label5 = tk.Label(root, text='Education\n(Graduate or Not Graduate): ')
canvas1.create_window(100, 220, window=label5)

entry5 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 220, window=entry5)

# New_Unemployment_Rate label and input box
label6 = tk.Label(root, text=' Self_Employed(Yes or No): ')
canvas1.create_window(100, 250, window=label6)

entry6 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 250, window=entry6)

label7 = tk.Label(root, text='ApplicantIncome(in hundreds): ')
canvas1.create_window(100, 280, window=label7)

entry7 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 280, window=entry7)

# New_Unemployment_Rate label and input box
label8 = tk.Label(root, text=' Coapplicant Income: ')
canvas1.create_window(100, 310, window=label8)

entry8 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 310, window=entry8)

label9 = tk.Label(root, text='Loan amount in thousands: ')
canvas1.create_window(100, 340, window=label9)

entry9 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 340, window=entry9)

# New_Unemployment_Rate label and input box
label10 = tk.Label(root, text=' Credit_history(1 or 0): ')
canvas1.create_window(100, 370, window=label10)

entry10 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 370, window=entry10)

label11= tk.Label(root, text='Loan_Term( in days): ')
canvas1.create_window(100, 400, window=label11)

entry11= tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 400, window=entry11)

# New_Unemployment_Rate label and input box
label12= tk.Label(root, text='Property Area\n(Urban, Semiurban, Rural): ')
canvas1.create_window(100, 430, window=label12)

entry12= tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 430, window=entry12)

def loan():
    global Loan_ID #our 1st input variable
    Loan_ID = str(entry1.get()) 
    
    global Gender	 #our 2nd input variable
    Gender	 = str(entry2.get()) 
    global Married #our 1st input variable
    Married = str(entry3.get()) 
    
    global Dependents #our 2nd input variable
    Dependents = str(entry4.get()) 
    global Education #our 1st input variable
    Education = str(entry5.get()) 
    
    global Self_Employed #our 2nd input variable
    Self_Employed = str(entry6.get()) 
    global ApplicantIncome #our 1st input variable
    ApplicantIncome = int(entry7.get()) 
    
    global CoapplicantIncome #our 2nd input variable
    CoapplicantIncome = float(entry8.get()) 
    global LoanAmount #our 1st input variable
    LoanAmount = float(entry9.get()) 
    
    global Loan_Amount_Term #our 2nd input variable
    Loan_Amount_Term = float(entry10.get()) 
    global Credit_History #our 1st input variable
    Credit_History = float(entry11.get()) 
    
    global Property_Area #our 2nd input variable
    Property_Area = str(entry12.get()) 
    global Loan_Status #our 2nd input variable
    Loan_Status = ""
    global Prediction_result #our 2nd input variable

    precleaned_data = pd.read_csv('loan_data_set.csv')
    a=[Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status]
    precleaned_data.loc[len(precleaned_data.index)] = a
    non_numeric = precleaned_data[['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']]
    numeric_data = precleaned_data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
    temp1 = non_numeric.groupby('Gender')['Loan_ID'].count().idxmax()
    temp2 = non_numeric.groupby('Married')['Loan_ID'].count().idxmax()
    temp3 = non_numeric.groupby('Dependents')['Loan_ID'].count().idxmax()
    temp4 = non_numeric.groupby('Self_Employed')['Loan_ID'].count().idxmax()
    non_numeric['Gender'] = non_numeric['Gender'].fillna(temp1)
    non_numeric['Married'] = non_numeric['Married'].fillna(temp2)
    non_numeric['Dependents'] = non_numeric['Dependents'].fillna(temp3)
    non_numeric['Self_Employed'] = non_numeric['Self_Employed'].fillna(temp4)
    mean1 = numeric_data['LoanAmount'].mean()
    mean2 = numeric_data['Loan_Amount_Term'].mean()
    mean3 = numeric_data['Credit_History'].mean()
    numeric_data['LoanAmount'] = numeric_data['LoanAmount'].fillna(mean1)
    numeric_data['Loan_Amount_Term'] = numeric_data['Loan_Amount_Term'].fillna(mean2)
    numeric_data['Credit_History'] = numeric_data['Credit_History'].fillna(mean3)
    encode_non_numeric = pd.get_dummies(non_numeric, columns = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'], drop_first = True)
    encode_non_numeric.drop('Loan_ID', axis = 1, inplace = True)
    new_loan_data = pd.concat([encode_non_numeric, numeric_data], axis = 1)
    x = new_loan_data.drop('Loan_Status_Y', axis = 1)
    y = new_loan_data['Loan_Status_Y']
    z = x.tail(1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    forest_model = RandomForestClassifier(n_estimators = 600)
    forest_model.fit(x_train, y_train)
    forest_y_pred = forest_model.predict(z)
    if forest_y_pred[0]==1:
        Prediction_result = 'Loan Approved'
    else:
        Prediction_result = 'Loan Disapproved'
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 500, window=label_Prediction)
button1 = tk.Button (root, text='Check Approval status',command=loan, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 460, window=button1)        
root.mainloop()