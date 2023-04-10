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

upperLimit, lowerLimit = LimitsCalculate(10)

population = populationInitialize(5, 10, upperLimit, lowerLimit)

visualizePopulation(population)

print(rouletteWheelSelection(population))