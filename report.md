# Project 3 - Deep Reinforcement Learning Report

- Group Name: Reinforcement
- Group Member:
    - z5129432: Yiqun Jiang
    - z5119193: Sichen Li
- Course: COMP9444 Assignment 3, Session 2, 2018
- Specification: https://www.cse.unsw.edu.au/~cs9444/18s2/hw3/index.html

## Batch size

test will terminated right after 2 minutes

Batch size      | Reward       
----------------|--------------------------------
4               | 94 -> 125 -> 131
8               | 132 -> 200
16              | 144 -> 200
32              | 200     
64              | 198     
128             | 151     
256             | 157     

## Episilon

episilon is the possibility of random choice

`episilon = epsilon - epsilon / decay`

episilon decay  | Reward
----------------|-----------------------------------------
1.1             | 9.2 -> 9.2 -> 8.8 -> 20.4 -> 28.0 -> 138     
1.2             | 21.9-> 120.1 -> 186
10              | 144 -> 200
33              | 20.2 -> 166 -> 122
100             | 111 -> 200
1000            | 172 -> 122 -> 117

## Gamma

gamma           | Reward      
----------------|-----------------------------------------------
0               | around 10
0.9             | 140 -> 127      
0.99            | 111 -> 200            
0.999           | 114 -> 150           
1               | 126-> 175

## Number of nodes in hidden layers 

hidden layer    | Reward      
----------------|--------------------------------
10              | around 10           
50              | 15 -> 30 -> 104 -> 113
100             | 111 -> 200            
1000            | 128.3            

## Finally we choose HyperParameters

Parameter       | Number
----------------|-------------
batch size      | 16
episilon decay  | 10
gamma           | 0.99    
hidden layers   | 100
