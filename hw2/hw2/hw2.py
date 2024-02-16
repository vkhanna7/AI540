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
    X=dict()
    
    with open (filename,encoding='utf-8') as f:
        # Add all characters 'a' to 'z' as keys with values set to 0
        for char in range(ord('A'), ord('Z')+1):
            X[chr(char)] = 0

        # Adding counts of all characters from file into the dict 
        for char in f.read():
            if char.isalpha():
                char = char.upper()
                if char in X:
                    X[char] = X[char] + 1
    f.close()
    
    #Formatting as required
    output = "Q1\n"
    
    for key in X: 
        if key == "Z":
            output += key + " " + str(X[key])
        else:
            output += key + " " + str(X[key]) + "\n"
    print(output)

    return X


def identify(letter):
    e, s = get_parameter_vectors()
    X = shred(letter)
    
    i  = X["A"]
    
    prob_e = "{0:.04f}".format(i*math.log(e[0]))
    prob_s = "{0:.04f}".format(i*math.log(s[0]))
    
    print("Q2\n" + prob_e + "\n" + prob_s)
    
    letters = []
    
    for char in range(ord('A'), ord('Z')+1):
            letters.append(chr(char))
            
    sum1 = 0.0000
    sum2 = 0.0000

    for x in range(26):
        sum1 += (X[letters[x]] * math.log(e[x]))
        sum2 += (X[letters[x]] * math.log(s[x]))    

    f_english = math.log(0.6) + sum1 
    f_spanish = math.log(0.4) + sum2 
    
    print("Q3\n" + "{0:.4f}".format(f_english) + "\n" + "{0:.4f}".format(f_spanish))
    
    p_english = 0.0000
    
    if f_spanish - f_english >= 100:
        p_english = 0.0000
    elif f_spanish - f_english <= -100:
        p_english = 1.0000
    else:
        p_english = (1/(1 + math.exp(f_spanish - f_english)))
        
    print("Q4\n" + "{0:.4f}".format(p_english) + "\n")
# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
#shred("letter0.txt")
identify("letter.txt")