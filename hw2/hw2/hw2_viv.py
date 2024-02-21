import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    
    with open (filename,encoding='utf-8') as file:
        # TODO: add your code here
        X = {"A": 0, "B": 0, "C": 0,"D": 0,"E": 0,"F": 0,"G": 0,"H": 0,"I": 0,"J": 
        0,"K": 0,"L": 0,"M": 0,"N": 0,"O": 0,"P": 0,"Q": 0,"R": 0,"S": 0,"T": 0,"U": 0,
         "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0}

         


         
        
        char_count = {}
        for line in file:
            line = line.upper()
            for char in line:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
        
        for key in X:
            if char_count.__contains__(key):
                X[key] = char_count[key]
        print(X)
        string = "Q1\n" 

        for key in X:
            if key.__eq__("Z"):
                string += key + " " + str(X[key])
            else:
                string += key + " " + str(X[key]) + "\n"

        print(string)

        return X

shred("letter0.txt")


def findLanguage():

    e,s = get_parameter_vectors()
    X = shred('letter.txt')
  
    i = X["A"]

    pe = "{0:.4f}".format(i * math.log(e[0]))

    ps = "{0:.4f}".format(i * math.log(s[0]))

    print("Q2\n" + pe + "\n" + ps)

    L = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    sum = 0.0000
    sum2 = 0.0000
    for x in range(26):
        sum += (X[L[x]] * math.log(e[x]))
        sum2 += (X[L[x]] * math.log(s[x]))


    fenglish = math.log(.6) + sum
    fspanish = math.log(.4) + sum2

    print("Q3\n" + "{0:.4f}".format(fenglish) + "\n" + "{0:.4f}".format(fspanish))
    
    penglish = 0.0000

    if fspanish - fenglish >= 100:
        penglish = 0.0000
    elif fspanish - fenglish <= -100:
        penglish = 1.0000
    else:
        penglish = (1/(1 + math.exp(fspanish - fenglish)))

    print("Q4\n" + "{0:.4f}".format(penglish) + "\n")


    


#findLanguage()



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!


