import os 
import numpy 
import pandas as pd
import seaborn as sns 

#Set working directory
os.getcwd()
os.chdir("/Users/sheeba/Desktop/data_science /python/Project8-PropertyPricePrediction/Dataset")

#Reading the data
 
predictionDf=pd.read_csv("Property_Price_Prediction.csv") #This is the prediction data 
#to check how the model interacts with unknown data

sourceDf=pd.read_csv("Property_Price_Train.csv") #Contains the Train Data + Test Data

#To see the print output properly
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
print(PredictionDf)

#Splitting the RawDf data into train and test with 80-20 ratio
from sklearn.model_selection import train_test_split
trainDf, testDf= train_test_split(sourceDf, random_state=7, shuffle=True)

#For easy identification of which random data is under which category
trainDf['Source']="Train"
testDf['Source']="Test"
predictionDf['Source']="Prediction"

#Combining the test+train+prediction data for easier identification & easy pre-processing
sourceDf=pd.concat([trainDf,testDf,predictionDf], axis=0)
sourceDf.shape


#Dropping 'ID' column that is useless for our analysis
sourceDf=sourceDf.drop(['Id'], axis=1)

#Checking for null values in the sourceDf data for missing value imputation [univariate analysis]

sourceDf.dtypes #checking the data type for missing value treatment

#loop method for missing value imputation [univariate analysis] 
sourceDf.isnull().sum()     

sourceDf_columns=sourceDf.columns
print(sourceDf_columns)
for i in (sourceDf_columns):
    
    if (i in ["Sale_Price", "Source"]):
        continue
    
    if trainDf[i].dtype == object:
        print("Cat: ", i)
        tempMode = sourceDf.loc[sourceDf["Source"] == "Train", i].mode()[0]
        sourceDf[i].fillna(tempMode, inplace = True)
    else:
        print("Cont: ", i)
        tempMedian = trainDf[i].median()
        sourceDf[i] = sourceDf[i].fillna(tempMedian)
     
        
sourceDf.isnull().sum()  #for checking       
# visual representation [bivariate analysis]

# bivariate analysis for continusous variables- scatterplot
corrDf=sourceDf[sourceDf["Source"]=="Train"].corr() #shows numbers
#corrDf.head() 
sns.heatmap(corrDf,linecolor="blue", xticklabels="auto", yticklabels="auto", cmap='winter')

# bivariate analysis for categorical variables- boxplot
categoricalVars=trainDf.columns[trainDf.dtypes==object]
print(categoricalVars)   
#running all of them and dumping it in a file for easier access


from matplotlib.backends.backend_pdf import PdfPages
filename="/Users/sheeba/Desktop/data_science /python/Project8-PropertyPricePrediction/Dataset/categoricalVars"
pdf=PdfPages(filename)        
for colNumber, colName in enumerate(categoricalVars): # enumerate gives key, value pair
    #print(colNumber, colName)
    figure()
    sns.boxplot(y = trainDf["Sale_Price"], x = trainDf[colName])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)

pdf.close()

counter = 0
for i in categoricalVars:
     print(i, counter)
     counter = counter + 1   
     
#dummy variable as python doesn't subsume categorical variables
sourceDf2=pd.get_dummies(sourceDf, drop_first=True) #drop_first for avoid perfect heteroscadicity

sourceDf2.shape
sourceDf.shape #check point

#adding the beta(0) or the intercept
from statsmodels.api import add_constant
sourceDf2=add_constant(sourceDf2)
sourceDf2.shape

#[sampling treatment]
# dividing sourceDf into train, test, and prediction
trainDf=sourceDf2[sourceDf2['Source_Train']==1].drop(['Source_Train', 'Source_Test'], axis=1).copy()
testDf=sourceDf2[sourceDf2['Source_Test']==1].drop(['Source_Train', 'Source_Test'], axis=1).copy()
predictionDf=sourceDf2[(sourceDf2['Source_Test']==0) & (sourceDf2['Source_Train']==0)].drop(['Source_Train', 'Source_Test'], axis=1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

#separating the independant variables (X[i])) and dependant variables (Y[i])
trainX=trainDf.drop(['Sale_Price'], axis=1).copy()
trainY=trainDf['Sale_Price'].copy()
testX=testDf.drop(['Sale_Price'], axis=1).copy()
testY=testDf['Sale_Price'].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape

#vif check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # The VIF that will be calculated at EVERY iteration in while loop
maxVIFCutoff = 5 # 5 is recommended cutoff value for linear regression
trainXCopy = trainX.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIFCutoff):
    
    print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    # tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = True)[-1:]["Column_Name"].values[0]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    # tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIFCutoff): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is intercept.
highVIFColumnNames

trainX = trainX.drop(highVIFColumnNames, axis = 1)
testX = testX.drop(highVIFColumnNames, axis = 1)
predictionDf = predictionDf.drop(highVIFColumnNames, axis = 1)

trainX.shape
testX.shape

#model-building
from statsmodels.api import OLS
m1ModelDef = OLS(trainY, trainX) # (Dep_Var, Indep_Vars) # This is model definition
m1ModelBuild = m1ModelDef.fit() # This is model building
m1ModelBuild.summary() 

dir(m1ModelBuild)
m1ModelBuild.pvalues

#model-optimisation
# Unlike linear regression in R, we dont have a "step()" function.
# We will use a for loop and discard indep variables based on "p-value"
# The concept of the for loop will remain very similar to VIF loop.

tempMaxPValue = 0.1
maxPValueCutoff = 0.1
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    
    print(counter)    
    
    tempModelDf = pd.DataFrame()    
    Model = OLS(trainY, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) # If there is some calculation error resulting in NAs
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames 

# Check final model summary
Model.summary()
trainX = trainX.drop(highPValueColumnNames, axis = 1)
testX = testX.drop(highPValueColumnNames, axis = 1)
predictionDf = predictionDf.drop(highPValueColumnNames, axis = 1)

trainX.shape
testX.shape

# Build model on trainX, trainY (after removing insignificant columns)
Model = OLS(trainY, trainX).fit()
Model.summary()

#model prediction
Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testY[:6]

#Model Diagnostic Plots (Validating the assumptions)

import seaborn as sns

# Homoskedasticity check
sns.scatterplot(Model.fittedvalues, Model.resid) # Should not show prominent non-constant variance (heteroskadastic) of errors against fitted values

# Normality of errors check
sns.distplot(Model.resid) # Should be somewhat close to normal distribution

# ## In case you would like the plots in different windows, you need to
# ## open a new window first using figure() method from matplotlib library
# ## figure() in python is equivalent to windows() in R
# from matplotlib.pyplot import figure
# figure()
# sns.scatterplot(Model.fittedvalues, Model.resid)
# figure()
# sns.distplot(Model.resid)

#########################
# Model Evaluation
#########################

# RMSE
import numpy as np
np.sqrt(np.mean((testY - Test_Pred)**2))
# This means on an "average", the house price prediction would have +/- error of about 56140

# Now, is this a good model? Probably an "Average" model. If I told you your house was going to sell for $300,000 and 
# then it actually only sold for $244,000 (Roughly $56,000 error), you would be pretty mad. $56,000 is a 
# reasonable difference when you're buying/selling a home. 
# if the prediction is $300000, then the house would be cold somewhere between $244000 and $356000 
# But what if I told you that I could predict GDP (Gross Domestic Product) 
# of the US with only an average error of $56000? Well, since the GDPs are usually around $20 trillion, 
# that difference (of $56,000) wouldn't be so big. So, an RMSE of $56,000 would be acceptable in a GDP model but not 
# in Property prediction model like ours! So, there is a bit of relativity involved in RMSE values.

# MAPE (Mean Absolute Percentage Error)
(np.mean(np.abs(((testY - Test_Pred)/testY))))*100
# This means on an "average", the house price prediction would have +/- error of 20%

# Generally, MAPE under 10% is considered very good, and anything under 20% is reasonable.
# MAPE over 20% is usually not considered great.

#########################
# Model Prediction (of prediction csv file)
#########################

predictionDf["Predicted_Sale_Price"] = Model.predict(predictionDf.drop(["Sale_Price"], axis = 1))

predictionDf.to_csv("predictionDf.csv")