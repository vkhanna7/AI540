import numpy as np 
import sys 
import matplotlib.pyplot as plt

def clean_data(file):
    start_year = 1855
    end_year = 2021
    raw_data = np.genfromtxt(file, delimiter=',', dtype=str, skip_header = 1)

    # Extracting years and days from Winter and Days of Ice columns 
    years = np.array([int(year.replace('"', '').split('-')[0]) for year in raw_data[:, 0]])
    #print(years)
   # Handling empty strings and non-numeric values in 'Days of Ice' column
    days = np.array([int(day.replace('"', '')) if day.replace('"', '').isdigit() else 0 for day in raw_data[:, -1]])
    #print(days)
    #Filtering required years 
    years_mask = (years >= start_year) & (years <= end_year)
    #print(years_mask)
    years = years[years_mask]
    days = days[years_mask]
    # Create a mask for unique years
    unique_mask = np.concatenate(([True], years[1:] != years[:-1]))

    # Use np.add.reduceat to sum the corresponding days for each unique year
    summed_days = np.add.reduceat(days, np.where(unique_mask)[0])

    # Create the final result by combining unique years and corresponding summed days
    result = np.column_stack((years[unique_mask], summed_days))
    #print(result)
    # Save the result into a CSV file
    header = "year,days"
    np.savetxt("hw5.csv", result, delimiter=',', header=header, comments='', fmt='%d')
    return result

def load_data(file):
    return np.genfromtxt(file, delimiter=',', dtype=str, skip_header = 1)


def visualize_data(data):
    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.title('Year vs Number of Frozen Days')
    plt.savefig('plot.jpg')
    #plt.show()

def getX(data):
    col1 = [1 for i in range(len(data))]
    result = np.column_stack((col1, data[:,0]))
    return np.array(result , dtype=np.int64)

def getY(data):
    return np.array(data[:,1], dtype=np.int64)

def getZ(X):
    X_transpose = np.transpose(X)
    X_transpose_dot_X = np.dot(X_transpose, X)
    return X_transpose_dot_X

def getI(Z):
    return np.linalg.inv(Z)

def getPI(X, I):
    return np.dot(I, X.transpose())

def getB(PI, Y):
    return np.dot(PI, Y)

def predict_ice_days(year, B):
    return str(B[0] + B[1] * year)

def compare(B):
    return  ">" if B > 0 else "<" if B < 0 else "="

def predict_x(num_days, B):
    return -B[0]/B[1] + num_days/B[1]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hw5.py filename.csv")
        sys.exit(1)

    filename = sys.argv[1]
    #cleaned_data = clean_data(filename)
    #print("Data saved as 'hw5.csv'")    
    cleaned_data = load_data(filename)
    visualize_data(cleaned_data)
    #print("Plot saved as 'plot.jpg'")
    X = getX(cleaned_data)
    print("Q3a:")
    print(X)

    Y = getY(cleaned_data)
    print("Q3b:")
    print(Y)

    Z = getZ(X)
    print("Q3c:")
    print(Z)

    I = getI(Z)
    print("Q3d:")
    print(I)

    PI = getPI(X, I)
    print("Q3e:")
    print(PI)

    B = getB(PI, Y)
    print("Q3f:")
    print(B)

    year = 2022
    predicted = predict_ice_days(year, B)
    print("Q4: " + predicted)

    print("Q5a: " + str(compare(np.sign(B[1]))))
    print("Q5b: A > (positive) sign indicates a direct relationship, as predictor variable B_1_hat increases so does number of ice days, \
            a < (negative) relationship means inverse relation such that as predictor variable increase, number of ice days decreases, \
            an = (0) relation means there is no linear correlation between the predictor variable and number of ice days.")
    
    print("Q6a: " + str(predict_x(0, B)))
    print("Q6b: Our model only takes into consideration past and current data points, it is not a compleling argument as it doesn't consider future climate changes and other factors")

    