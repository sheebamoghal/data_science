import os
import pandas as pd
import numpy as np
import seaborn as sns

# Increase the print output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set working directory
os.chdir("C:/Users/rahul/Documents/IMR/Data_Linear_Regression/")

# Read in the data
rawDf = pd.read_csv("PropertyPrice_Data.csv")
predictionDf = pd.read_csv("PropertyPrice_Prediction.csv")


# Before we do anything, we need to divide our Raw Data into train and test sets (validation)
from sklearn.model_selection import train_test_split
trainDf, testDf = train_test_split(rawDf, train_size=0.8, random_state = 150)

# Create Source Column in both Train and Test
trainDf['Source'] = "Train"
testDf['Source'] = "Test"
predictionDf['Source'] = "Prediction"

# Combine Train and Test
fullRaw = pd.concat([trainDf, testDf, predictionDf], axis = 0)
fullRaw.shape


# Lets drop "Id" column from the data as it is not going to assist us in our model
fullRaw = fullRaw.drop(['Id'], axis = 1) 
# fullRaw.drop(['Id'], axis = 1, inplace = True) 

# Check for NAs
fullRaw.isnull().sum()

# Check data types of the variables
fullRaw.dtypes



############################
# Univariate Analysis: Missing value imputation
############################

# Garage variable (Categorical)
tempMode = fullRaw.loc[fullRaw["Source"] == "Train", "Garage"].mode()[0] # Inference (Based on Trainset)
# tempMode = trainDf["Garage"].mode()[0]

fullRaw["Garage"].fillna(tempMode, inplace = True) # Action (Based on Fullraw)     
# fullRaw["Garage"] = fullRaw["Garage"].fillna(tempMode)  

# Garage_Built_Year (Continuous)
# tempMedian = fullRaw.loc[fullRaw["Source"] == "Train", "Garage_Built_Year"].median()
tempMedian = trainDf["Garage_Built_Year"].median() # Same as above
tempMedian
fullRaw["Garage_Built_Year"] = fullRaw["Garage_Built_Year"].fillna(tempMedian)   

# All NAs should be gone now
fullRaw.isnull().sum()

# Task: Attempt writing an automated for loop code to automate missing value imputation (Just like in R)

############################
# Bivariate Analysis Continuous Variables: Scatterplot
############################

corrDf = fullRaw[fullRaw["Source"] == "Train"].corr()
# corrDf.head()
sns.heatmap(corrDf, 
        xticklabels=corrDf.columns,
        yticklabels=corrDf.columns, cmap='YlOrBr')

# # To know possible names of color palettes, pass an incorrect palette name (cmap argument) in the below code:
# sns.heatmap(corrDf, 
#         xticklabels=corrDf.columns,
#         yticklabels=corrDf.columns, cmap='wqeweqeqw')


############################
# Bivariate Analysis Categorical Variables: Boxplot
############################

categoricalVars = trainDf.columns[trainDf.dtypes == object]
categoricalVars

# First option: Do it manually for each variable
sns.boxplot(y = trainDf["Sale_Price"], x = trainDf["Road_Type"])
sns.boxplot(y = trainDf["Sale_Price"], x = trainDf["Property_Shape"])

# Second option: Run a for loop and create multiple plots
from matplotlib.pyplot import figure
for colName in categoricalVars:
    figure()    
    sns.boxplot(y = trainDf["Sale_Price"], x = trainDf[colName])

# Third option: Run a for loop to dump all the plots in a pdf file
from matplotlib.backends.backend_pdf import PdfPages
fileName = "C:/Users/rahul/Documents/IMR/Data_Linear_Regression/Categorical_Variables_Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categoricalVars): # enumerate gives key, value pair
    # print(colNumber, colName)
    figure()
    sns.boxplot(y = trainDf["Sale_Price"], x = trainDf[colName])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)
    
pdf.close()

# counter = 0=
# for i in categoricalVars:
#     print(i, counter)
#     counter = counter + 1

############################
# Dummy variable creation
############################

fullRaw2 = pd.get_dummies(fullRaw, drop_first = True)  # drop_first = True will ensure you get n-1 dummies for each categorcial var
# 'Source'  column will change to 'Source_Train' & 'Source_Test' and it contains 0s and 1s

fullRaw2.shape
fullRaw.shape

# Start next session talking about Why shouldnt we follow R's automatic dummy creation in the model.

############################
# Add Intercept Column
############################

# In Python, linear regression function does NOT account for an intercept by default.
# So, we need to explicitely add intercept (in the df) - a column called "const" with all values being 1 in it.
from statsmodels.api import add_constant
fullRaw2 = add_constant(fullRaw2)
fullRaw2.shape

############################
# Sampling
############################

# Step 1: Divide into Train, Test and Prediction Dfs
trainDf = fullRaw2[fullRaw2['Source_Train'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
testDf = fullRaw2[fullRaw2['Source_Test'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
predictionDf = fullRaw2[(fullRaw2['Source_Train'] == 0) & 
                        (fullRaw2['Source_Test'] == 0)].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Step 2: Divide into Xs (Indepenedents) and Y (Dependent)
trainX = trainDf.drop(['Sale_Price'], axis = 1).copy()
trainY = trainDf['Sale_Price'].copy()
testX = testDf.drop(['Sale_Price'], axis = 1).copy()
testY = testDf['Sale_Price'].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape


#########################
# VIF check
#########################

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


#########################
# Model Building
#########################

from statsmodels.api import OLS
m1ModelDef = OLS(trainY, trainX) # (Dep_Var, Indep_Vars) # This is model definition
m1ModelBuild = m1ModelDef.fit() # This is model building
m1ModelBuild.summary() # This is model output summary



# # Some places, model def and model building are comibined together like below:
# m1ModelBuild = OLS(trainY, trainX).fit()
# m1ModelBuild.summary() # This is model output summary


# 39.33 is the BsmtFinSF1 Coeff, 4.200 is the Std Error (Found in model output table) 
# 2.5% Up and Down (5% in total) for BsmtFinSF1 Coeff is calculated as:
# Lower 2.5% is (39.33 - 1.96 * 4.20) -> 31.09 & Upper 2.5% is (39.33 + 1.96 * 4.20) -> 47.5
# 1.96 is the 95% Confidence Interval (CI)

# Extract/ Identify p-values from model
dir(m1ModelBuild)
m1ModelBuild.pvalues

#########################
# Model Optimization
#########################

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

#########################
# Model Prediction
#########################


Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testY[:6]

#########################
# Model Diagnostic Plots (Validating the assumptions)
#########################

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



