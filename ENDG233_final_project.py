"""
endg233_final_project.py
LUJAINA ELDELEBSHANY, ROMANARD TIRATIRA ENDG 233 F21
A terminal-based application to process, plot, and predict data based on given user input and provided csv files.
Built-in functions, classes, and models that support compound data structures, user entry, and casting are used.
"""

#library imports
import matplotlib.pyplot as plt

#logistic regression imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class Predict:
    '''A class used to create a predict object. Instance method runs a prediction algorithm based on logistic regression
     using parameters age, resting blood pressure, cholesterol, and maximum heart rate.  
        Parameters: None
        Return: none
        Attributes:
            age(int): int that represents the patient/users age
            sex(str): string that represent the patients/users gender
            resting_BP(int): int that represents the patients/users resting blood pressure
            cholesterol(int): int that represents the patients/users cholesterol
            max_HR(int): int that represents the max heart rate of the patient/user
            chest_pain_type(str): string that represents the type of chest paint the patient/user has
    '''
    def __init__(self, age = None, sex = None, resting_BP = None, cholesterol = None, max_HR = None, chest_pain_type = None):
        self.chest_pain_type = chest_pain_type 
        self.age = age
        self.resting_BP = resting_BP
        self.cholesterol = cholesterol
        self.sex = sex
        self.max_HR = max_HR

    #prediction model
    def predict_if_heart_disease(self):

        ATA_data_predict = pd.read_csv('final_project_data_ATA.csv')
        NAP_data_predict = pd.read_csv('final_project_data_NAP.csv')
        ASY_data_predict = pd.read_csv('final_project_data_ASY.csv')

        #independant/dependant variable assignment and split into train and test at 70:30 ratio
        X_ATA = ATA_data_predict[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
        Y_ATA = ATA_data_predict['HeartDisease']

        X_NAP = NAP_data_predict[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
        Y_NAP = NAP_data_predict['HeartDisease']

        X_ASY = ASY_data_predict[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
        Y_ASY = ASY_data_predict['HeartDisease']

        X_train_ATA,X_test_ATA,y_train_ATA,y_test_ATA = train_test_split(X_ATA,Y_ATA,test_size=0.3,random_state=0)  
        X_train_NAP,X_test_NAP,y_train_NAP,y_test_NAP = train_test_split(X_NAP,Y_NAP,test_size=0.3,random_state=0)  
        X_train_ASY,X_test_ASY,y_train_ASY,y_test_ASY = train_test_split(X_ASY,Y_ASY,test_size=0.3,random_state=0)  
                
        #creates and fit models for each chest pain type dataset
        log_regression1 = LogisticRegression()
        log_regression2 = LogisticRegression()
        log_regression3 = LogisticRegression()
                
        log_regression1.fit(X_train_ATA,y_train_ATA)
        log_regression2.fit(X_train_NAP,y_train_NAP)
        log_regression3.fit(X_train_ASY,y_train_ASY)

        y_pred1 = log_regression1.predict(X_test_ATA)
        y_pred2 = log_regression2.predict(X_test_NAP)
        y_pred3 = log_regression3.predict(X_test_ASY)

        #split_ypred1 refers to that moment's prediction for that value
        split_y_pred1 = np.array_split(y_pred1, len(y_pred1))
        split_y_pred2 = np.array_split(y_pred2, len(y_pred2))
        split_y_pred3 = np.array_split(y_pred3, len(y_pred3))

        # reshape array into right dimension and turn into expected numpy array
        input_array = np.array([self.age, self.resting_BP, self.cholesterol, self.max_HR])
        reshaped_array = input_array.reshape(1,-1)

        #run correct model based on file of data referenced with pain type
        if self.chest_pain_type == 'ATA':
            model_result = log_regression1.predict(reshaped_array)
        elif self.chest_pain_type == 'ASY':
            model_result = log_regression2.predict(reshaped_array)
        elif self.chest_pain_type == 'NAP':
            model_result = log_regression3.predict(reshaped_array)
        else:
            return 'Sorry, your input is not supported.'
        
        #prints results of model
        if model_result == 1:
            print('Your patient is predicted to suffer from heart failure.')
        elif model_result == 0:
            print('Your patient is predicted to not suffer from heart failure.')
        else:
            print('Sorry, try again later.')

def average_BP(patient):
    '''Function that computes the average BP from the user inputed age and chest pain.
            Parameters: patient(Predict): Predict object containing patient information
            Returns: Average BP of the specific age and chest pain input
    '''
    age_BP_matches = []
    if patient.chest_pain_type == 'non-asymptomatic':
        data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
        age_BP_matches = [int(data_NAP[i][2]) for i in range(199) if int(data_NAP[i][0]) == patient.age]
    elif patient.chest_pain_type == 'asystol':
        data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
        age_BP_matches = [int(data_ASY[i][2]) for i in range(199) if int(data_ASY[i][0]) == patient.age]
    elif patient.chest_pain_type == 'arterial tachycardia':
        data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
        age_BP_matches = [int(data_ATA[i][2]) for i in range(199) if int(data_ATA[i][0]) == patient.age]
    return round((sum(age_BP_matches)/len(age_BP_matches)), 2)

def average_age(patient):
    '''Function that computes the average BP from the user inputed age, sex and chest pain.
            Parameters: 
                patient(Predict): Predict object containing patient information
            Returns: Average BP of the specific age, sex, and chest pain of patient
    '''
    if patient.chest_pain_type == 'non-asymptomatic':
        data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
        gender_BP_matches = [int(data_NAP[i][0]) for i in range(199) if data_NAP[i][1] == patient.sex and int(data_NAP[i][2]) == patient.resting_BP]
    elif patient.chest_pain_type == 'asystol':
        data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
        gender_BP_matches = [int(data_ASY[i][0]) for i in range(199) if data_ASY[i][1] == patient.sex and int(data_ASY[i][2]) == patient.resting_BP]
    elif patient.chest_pain_type == 'arterial tachycardia':
        data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
        gender_BP_matches = [int(data_ATA[i][0]) for i in range(199) if data_ATA[i][1] == patient.sex and int(data_ATA[i][2]) == patient.resting_BP]
    if(len(gender_BP_matches) == 0):
        return 'No matching data'
    return round((sum(gender_BP_matches)/len(gender_BP_matches)),2)

def percentage_of_heart_failure(patient):
    '''Calculates the percentage of people with a specific age and gender with heart disease across all chest pains.
            Parameters:
                patient(Predict): Predict object containing patient information
            Returns: The percentage of people with heart disease with a specific age and gender
    '''
    data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
    data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
    data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
    gender_sex_matches_NAP = [int(data_NAP[i][5]) for i in range(199) if data_NAP[i][1] == patient.sex and int(data_NAP[i][0]) == patient.age]
    gender_sex_matches_ASY = [int(data_ASY[i][5]) for i in range(199) if data_ASY[i][1] == patient.sex and int(data_ASY[i][0]) == patient.age]
    gender_sex_matches_ATA = [int(data_ATA[i][5]) for i in range(199) if data_ATA[i][1] == patient.sex and int(data_ATA[i][0]) == patient.age]
    gender_sex_matches = gender_sex_matches_NAP + gender_sex_matches_ASY + gender_sex_matches_ATA
    if len(gender_sex_matches) == 0:
        return 'No matching data'
    return round(((sum(gender_sex_matches)/len(gender_sex_matches)) * 100), 2)


def minimum_age(patient):
    '''Find the minimum age in which a person gets heart disease with a specific BP and chest pain
            Parameters:
                patient(Predict): Predict object containing patient information
            Returns: The minimum age for an idividual to get heart disease with specific BP and chest pain
    '''
    if patient.chest_pain_type != 'non-asymptomatic':
        data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_BP_matches = [data_NAP[i][0] for i in range(199) if int(data_NAP[i][2]) == patient.resting_BP and int(data_NAP[i][5]) == 1]
    elif patient.chest_pain_type == 'asystol':
        data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_BP_matches = [int(data_ASY[i][0]) for i in range(199) if int(data_ASY[i][2]) == patient.resting_BP and int(data_ASY[i][5]) == 1]
    elif patient.chest_pain_type == 'arterial tachycardia':
        data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_BP_matches = [int(data_ATA[i][0]) for i in range(199) if int(data_ATA[i][2]) == patient.resting_BP and int(data_ATA[i][5]) == 1]
    return min(chest_pain_BP_matches)

def minimum_BP(age, chest_pain):
    '''Find the minimum BP in which a person gets heart disease with a specific age and chest pain
            Parameters:
                age(int): int that represents the patient/user age
                chest_pain(str): string that represents that patient/user chest pain
            Returns: The minimum BP for an idividual to get heart disease with specific age and chest pain
    '''
    if chest_pain == 'non-asymptomatic':
        data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_NAP[i][2]) for i in range(199) if int(data_NAP[i][0]) == age and int(data_NAP[i][5]) == 1]
    elif chest_pain == 'asystol':
        data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_ASY[i][2]) for i in range(199) if int(data_ASY[i][0]) == age and int(data_ASY[i][5]) == 1]
    elif chest_pain == 'arterial tachycardia':
        data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_ATA[i][2]) for i in range(199) if int(data_ATA[i][0]) == age and int(data_ATA[i][5]) == 1]
    if(len(chest_pain_age_matches) == 0):
        return np.nan
    return min(chest_pain_age_matches)

def average_cholesterol(age, chest_pain):
    '''Calculate the average cholesterol of people of a specific age
            Parameters:
                age(int): int that represents the patient/user age
                chest_pain(str): string that represents that patient/user chest pain
            Returns: The average cholesterol of people of a certain age
    '''
    if chest_pain == 'non-asymptomatic':
        data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_NAP[i][3]) for i in range(199) if int(data_NAP[i][0]) == age]
    elif chest_pain == 'asystol':
        data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_ASY[i][3]) for i in range(199) if int(data_ASY[i][0]) == age]
    elif chest_pain == 'arterial tachycardia':
        data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)
        chest_pain_age_matches = [int(data_ATA[i][3]) for i in range(199) if int(data_ATA[i][0]) == age]
    return sum(chest_pain_age_matches)/len(chest_pain_age_matches)

def main():

    data_ASY = np.genfromtxt('final_project_data_ASY.csv', delimiter = "," , skip_header = True, dtype = str)
    data_NAP = np.genfromtxt('final_project_data_NAP.csv', delimiter = "," , skip_header = True, dtype = str)
    data_ATA = np.genfromtxt('final_project_data_ATA.csv', delimiter = "," , skip_header = True, dtype = str)

    #introducing the program, and taking valid user input
    print('Heart Disease Predictor and Data Analyzer')
    print()

    input_age = int(input('Please enter the age of the patient: '))
    while input_age > 74 or input_age < 28:
        input_age = int(input('Your age is not supported by this data, please enter a valid age between 28 and 74 years of age: '))
    print()

    input_sex = input('Please enter the sex of the patient (M/F): ')
    while input_sex != 'M' and input_sex !='F': 
        input_sex = input('Please re enter a valid sex, either "M" for male or "F" for female: ')
    print()

    input_restBP = int(input('Please enter the resting blood pressure of the patient: '))
    while input_restBP < 0: 
        input_restBP = int(input('Your number was invalid, re-enter a rest blood pressure within a reasonable range: '))
    print()

    input_cholesterol = int(input('Please enter the cholesterol of the patient: '))
    while input_cholesterol < 0: 
        input_cholesterol = int(input('Your number was invalid, re-enter a cholesterol input within a reasonable range: '))
    print()
    
    input_max_HR = int(input('Please enter the maximum heart rate of the patient: '))
    while input_max_HR < 0:
        input_max_HR = int(input('Your number was invalid, re-enter a rest blood pressure within a reasonable range: '))
    print()
    
    input_pain_type = input('Please enter the type of chest pain the patient is experiencing, (either non-asymptomatic, asystole, or arterial tachycardia): ')
    while input_pain_type != 'non-asymptomatic' and input_pain_type != 'asystole' and input_pain_type != 'arterial tachycardia':
        input_pain_type = input('Your input was invalid, re-enter a type of chest pain as previously suggested: ')
    print()
    
    #sets object with all user input as properties
    patient = Predict(input_age, input_sex, input_restBP, input_cholesterol, input_max_HR, input_pain_type)
    
    #take user's input for which function to run, runs infinitely by choice until exit
    while(True):
        print('1. Predict if heart disease will happen based on stored data')
        print('2. Average blood pressure for a patient with similar age and diagnosis conditions')
        print('3. Average age for patients with selected type of user pain')
        print('4. Statistical data on frequency of those with a specific age and sex to suffer from heart failure, including non-symptomatic percentage')
        print('5. Minimum age for  patient that has suffered heart failure with with a blood pressure match')
        menu_choice = int(input('Please select one of the following choices from the menu of program options: '))
        while menu_choice != 1 and menu_choice != 2 and menu_choice != 3 and menu_choice != 4 and menu_choice != 5:
            menu_choice = int(input('Please enter a VALID option from the menu numbers: '))
        print()
        if(menu_choice == 1):
            if input_pain_type == 'non-asymptomatic':
                patient = Predict(input_age, input_sex, input_restBP, input_cholesterol, input_max_HR, 'NAP')
                patient.predict_if_heart_disease()
            elif input_pain_type == 'asystole':
                patient = Predict(input_age, input_sex, input_restBP, input_cholesterol, input_max_HR, 'ASY')                
                patient.predict_if_heart_disease()
            elif input_pain_type == 'arterial tachycardia':
                patient = Predict(input_age, input_sex, input_restBP, input_cholesterol, input_max_HR, 'ATA')               
                patient.predict_if_heart_disease()
        elif (menu_choice == 2):
            print(f'{average_BP(patient)} is the average BP of those your age.')
        elif (menu_choice == 3):
            print(f'The average age of a patients with {input_pain_type} pain is {average_age(patient)} years old.')
        elif (menu_choice == 4):
            print(f'{percentage_of_heart_failure(patient)}% of people of your age gender and gender have heart disease.')
        elif (menu_choice == 5):
            print(f'The youngest recorded age for a patient with the input blood pressure is {minimum_age(patient)} years old.')

        print('Would you like to calculate more statistics? (Y/N)')
        menu_replay = input()
        if menu_replay == 'N':
            break

    #figure x-values
    plot_age_ASY = list({int(data_ASY[i][0]) for i in range(199)})
    plot_age_ASY.sort()
    plot_age_NAP = list({int(data_NAP[i][0]) for i in range(199)})
    plot_age_NAP.sort()
    plot_age_ATA = list({int(data_ATA[i][0]) for i in range(199)})
    plot_age_ATA.sort()

    #figure 1
    plt.plot(plot_age_ASY, [minimum_BP(plot_age_ASY[i], 'asystol') for i in range(len(plot_age_ASY))], 'b:', label = 'Asystol')
    plt.plot(plot_age_ATA, [minimum_BP(plot_age_ATA[i], 'arterial tachycardia') for i in range(len(plot_age_ATA))], 'm:', label = 'Arterial tachycardia')
    plt.plot(plot_age_NAP, [minimum_BP(plot_age_NAP[i], 'non-asymptomatic') for i in range(len(plot_age_NAP))], 'y:', label = 'Non-Asymptomatic')
    plt.title('Minimum Blood Pressure for Heart Disease by Age')
    plt.xlabel('Age in Years')
    plt.ylabel('Blood Pressure')
    plt.legend(loc = 'upper left', shadow = True)
    plt.show()

    #figure 2
    plt.subplot(3, 1, 1)
    plt.plot(plot_age_ASY, [average_cholesterol(plot_age_ASY[i], 'asystol') for i in range(len(plot_age_ASY))], 'b:', label = 'Asystol')
    plt.legend(loc ='upper right', shadow = True)
    plt.title('Average Cholesterol by Age')
    plt.ylabel('Cholesterol')
    plt.subplot(3, 1, 2)
    plt.plot(plot_age_ATA, [average_cholesterol(plot_age_ATA[i], 'arterial tachycardia') for i in range(len(plot_age_ATA))], 'm:', label = 'Arterial tachycardia')
    plt.legend(loc ='bottom left', shadow = True)
    plt.ylabel('Cholesterol')
    plt.subplot(3, 1, 3)
    plt.plot(plot_age_NAP, [average_cholesterol(plot_age_NAP[i], 'non-asymptomatic') for i in range(len(plot_age_NAP))], 'y:', label = 'Non-asymptomatic')
    plt.legend(loc ='bottom left', shadow = True)
    plt.ylabel('Cholesterol')
    plt.xlabel('Age in Years')
    plt.show()

    print('Thank you for using the program.')

if __name__ == '__main__':
    main()