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
    def createFitnessValuesGenesList (self, callback):
        return [callback(gene) for gene in self.genes]
    def calculateFitnessValueOfTheChromosome(self, callback):
        return sum(map(fitnessFuction, self.genes))

def LimitsCalculate(numberOfGenes):
    upperLimit = np.ones(numberOfGenes, dtype=int) * 100
    lowerLimit = np.ones(numberOfGenes, dtype=int) * (-100)
    return [upperLimit, lowerLimit]
#chosen just for this case
#this is the space for your solution, the limits of the values of your Chromosome's genes

def visualizePopulation(population):
    for i in range(len(population)):
        print("Chromosome " , i)
        print(population[i].genes)

def populationInitializate(numberOfChromosomes, numberOfGenes, upperLimits, lowerLimits):
    population = []
    for i in range(numberOfChromosomes):
        population.append(Chromosome([]))
    for i in range(numberOfChromosomes):
        for j in range(numberOfGenes):
            population[i].genes.append((upperLimits[j] - lowerLimits[j]) * rd.random() + lowerLimits[j])
    return population


upperLimit, lowerLimit = LimitsCalculate(10)

population = populationInitializate(2, 10, upperLimit, lowerLimit)

visualizePopulation(population)