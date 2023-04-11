import random as rd
import math
import sys
import numpy as np


def fitnessFuction(x):
    return x*x
    # you can modify to whatever function you want, here it's just an example
    # it calculates the square for any number

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
    def calculateFitnessValueOfTheChromosome(self, callback):
        return sum(map(callback, self.genes))

def LimitsCalculate(numberOfGenes, up=100, lp=-100):
    upperLimit = np.ones(numberOfGenes, dtype=int) * up
    lowerLimit = np.ones(numberOfGenes, dtype=int) * lp
    return [upperLimit, lowerLimit]
#chosen just for this case
#this is the space for your solution, the limits of the values of your Chromosome's genes

def visualizePopulation(population):
    for i in range(len(population)):
        print("Chromosome " , i)
        print(population[i].genes)

def populationInitialize(numberOfChromosomes, numberOfGenes, upperLimits, lowerLimits):
    population = []
    for i in range(numberOfChromosomes):
        population.append(Chromosome([]))
    for i in range(numberOfChromosomes):
        for j in range(numberOfGenes):
            population[i].genes.append((upperLimits[j] - lowerLimits[j]) * rd.random() + lowerLimits[j])
    return population

def rouletteWheelSelection(population):
    populationOrderedByFitness = sorted(population,
                                        key=lambda p: p.calculateFitnessValueOfTheChromosome(fitnessFuction),
                                        reverse=True)
    print("Chromosomes Adresses ordered by their fitness:")
    print(populationOrderedByFitness)
    fitnessList = list(map(lambda p: p.calculateFitnessValueOfTheChromosome(fitnessFuction), population))
    cumSum = list(np.cumsum([c/sum(fitnessList) for c in fitnessList]))
    cumSum.sort(reverse=True)
    print("\n Cumulative sum in descending order:" )
    print(cumSum)
    R  =  rd.random()
    parent1Index = len(cumSum) - 1 #guarantee that it has an index if we can't be in the if condition
    for i in range(len(cumSum)):
        if R > cumSum[i]:
            parent1Index = i-1
            break
    parent2Index = parent1Index
    while parent2Index == parent1Index:
        R = rd.random()
        for i in range(len(cumSum)):
            if R>cumSum[i]:
                parent2Index = i - 1;
                break
    parent1 = populationOrderedByFitness[parent1Index]
    parent2 = populationOrderedByFitness[parent2Index]
    return [parent1, parent2]

def crossover(parent1, parent2, crossoverProbability, crossoverType):
    match crossoverType:
        case "single":
            ub = len(parent1.genes) - 1
            lb = 0
            Cross_P = round ((ub-lb) * rd.random() + lb)
            #We have to be sure that cross_P is not 0 or the upper limit
            while(Cross_P == 0 or Cross_P == ub ):
                Cross_P = round((ub-lb) * rd.random() + lb)
            part1FromChild1 = parent1.genes[0:Cross_P]
            part2FromChild1 = parent2.genes[Cross_P:]
            child1 = Chromosome(part1FromChild1+part2FromChild1)
            part1FromChild2 = parent2.genes[0:Cross_P]
            part2FromChild2 = parent1.genes[Cross_P:]
            child2 = Chromosome(part1FromChild2+part2FromChild2)
        case 'double':
            ub = len(parent1.genes) - 1
            lb = 0
            Cross_P1 = round((ub - lb) * rd.random() + lb)
            while(Cross_P1 == 0 or Cross_P1 == ub ):
                Cross_P1 = round((ub-lb) * rd.random() + lb)
            Cross_P2 = Cross_P1
            while(Cross_P2 == Cross_P1 or Cross_P2 == 0 or Cross_P2 == ub):
                Cross_P2 = round((ub - lb) * rd.random() + lb)
            if Cross_P1 > Cross_P2:
                [Cross_P1, Cross_P2] = [Cross_P2,Cross_P1]
            part1FromChild1 = parent1.genes[0:Cross_P1]
            part2FromChild1 = parent2.genes[Cross_P1:Cross_P2]
            part3FromChild1 = parent1.genes[Cross_P2:]
            child1 = Chromosome(part1FromChild1 + part2FromChild1 + part3FromChild1)
            part1FromChild2 = parent2.genes[0:Cross_P1]
            part2FromChild2 = parent1.genes[Cross_P1:Cross_P2]
            part3FromChild2 = parent2.genes[Cross_P2:]
            child2 = Chromosome(part1FromChild2 + part2FromChild2 + part3FromChild2)
    R1 = rd.random()
    if R1<=crossoverProbability:
        child1 = child1
    else:
        child1 = parent1
    R2 = rd.random()
    if R2<=crossoverProbability:
        child2 = child2
    else:
        child2 = parent2
    return [child1, child2]









upperLimit, lowerLimit = LimitsCalculate(10)

population = populationInitialize(5, 10, upperLimit, lowerLimit)

visualizePopulation(population)

[parent1, parent2] = rouletteWheelSelection(population)
print("PARENT 1")
print(parent1.genes)
print("PARENT 2")
print(parent2.genes)


[child1, child2] = crossover(parent1, parent2, 0.85, "double")

print("CHILD 1")
print(child1.genes)
print("CHILD 2")
print(child2.genes)
