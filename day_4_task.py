# -*- coding: utf-8 -*-
"""Day-4_task.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DfFHPhCQ4Rcpxi_4QBoo1jYSfvZa8CYr
"""

#Write a Python program to calculate the sum of all even numbers between 1 and a given positive integer n
a=int(input("enter ending value: "))
sum=0
print("The even numbers between 1 and a given positive integer ",a,"is: ")
for i in range(1,a+1):
  if i%2==0:
    print(i,end=",")
    sum+=i
print("\n the sum of even numbers between 1 and positive integer",a,"is:",sum)