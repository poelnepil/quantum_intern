# You have a positive integer number N as an input.
# Please write a program in Python 3 that calculates the sum in range 1 and N.

n = int(input())
print(sum([int(i) for i in range(n+1)]))