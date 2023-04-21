import random as rd
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def fitnessFunction(x):
    return x*x
    # you can modify to whatever function you want, here it's just an example
    # it calculates the square for any number

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
    def calculateFitnessValueOfTheChromosome(self, callback):
        return sum(map(callback, self.genes))

def visualizePopulation(population):
    for i in range(len(population)):
        print("Chromosome " , i+1)
        print(population[i].genes)

def populationInitialize(numberOfChromosomes, numberOfGenes):
    population = []
    x = (0,1)
    for i in range(numberOfChromosomes):
        population.append(Chromosome([]))
    for i in range(numberOfChromosomes):
        for j in range(numberOfGenes):
            population[i].genes.append(rd.choice(x))
    return population

def rouletteWheelSelection(population, callback):
    populationOrderedByFitness = sorted(population,
                                        key=lambda p: p.calculateFitnessValueOfTheChromosome(callback),
                                        reverse=True)
    fitnessList = list(map(lambda p: p.calculateFitnessValueOfTheChromosome(callback), population))
    cumSum = list(np.cumsum([c/sum(fitnessList) for c in fitnessList]))
    cumSum.sort(reverse=True)
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

def mutation(child, mutationProbability):
    for i in range(len(child.genes)):
        R = rd.random()
        if R < mutationProbability:
            child.genes[i] = int (not child.genes[i])
    return child

def elitism(population, newPopulation, elitismRate, callback):
    elitismNumber = round (len(newPopulation) * elitismRate)
    #newPopulation is the population after mutation process
    oldestPopulationOrderedByFitness = sorted(population,
                                        key=lambda p: p.calculateFitnessValueOfTheChromosome(callback),
                                        reverse=True)
    newestPopulationOrderedByFitness = sorted(newPopulation,
                                             key=lambda p: p.calculateFitnessValueOfTheChromosome(callback),
                                             reverse=True)
    for i in range(elitismNumber):
        newPopulation[i] = oldestPopulationOrderedByFitness[i]
    for j in range(elitismNumber, len(newPopulation)):
        newPopulation[j] = newestPopulationOrderedByFitness[j-elitismNumber]
    return newPopulation

def binaryGeneticAlgorithm(numberOfChromosomes, numberOfGenes, numberOfGenations, crossoverProbability, crossoverType,
                               mutationProbability, elitismRate, fitnessFunction):
    population = populationInitialize(numberOfChromosomes, numberOfGenes)
    visualizePopulation(population)
    print(" BEST CHROMOSOME OF THE FIRST GENERATION ")
    print(max(map(lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction),population)))
    generations = [i+1 for i in range(numberOfGenations)]
    bestChromosomesFitnessValues = []
    bestChromosomesFitnessValues.append(max(map(lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction),population)))
    bestChromosomesList = list(filter(lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction) ==
                                 max(map(lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction),population)), population))
    y_axis = [bestChromosomesFitnessValues[0]]
    for j in range(1,numberOfGenations):
        newPopulation = []
        for k in range(0,numberOfChromosomes//2):
            [parent1, parent2] = rouletteWheelSelection(population, fitnessFunction)
            [child1, child2] = crossover(parent1, parent2, crossoverProbability, crossoverType)
            child1 = mutation(child1, mutationProbability)
            child2 = mutation(child2, mutationProbability)
            newPopulation.append(child1)
            newPopulation.append(child2)
        newPopulation = elitism(population, newPopulation, elitismRate, fitnessFunction)
        bestChromosomesFitnessValues.append(
            max(map(lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction), newPopulation)))
        bestChromosomesList.append(sorted(newPopulation, key=lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction),reverse=True)[0])
        y_axis.append(max(bestChromosomesFitnessValues))
    print("\n----------------------")
    print("Best Fitness Value is")
    bestChromosome = sorted(bestChromosomesList, key=lambda c: c.calculateFitnessValueOfTheChromosome(fitnessFunction))[len(bestChromosomesList) - 1]
    print(bestChromosome.calculateFitnessValueOfTheChromosome(fitnessFunction))
    print("\n----------------------")
    print("The best chromosome has the following genes")
    print(bestChromosome.genes)
    font1 = {'family':'serif','color':'blue','size':15}
    font2 = {'family':'serif','color':'darkred','size': 10}
    plt.subplot(2,2,1)
    plt.plot(generations, y_axis, c = '#4CAF50', linewidth = '3.5')
    plt.title("Best chromosome fitness value per generation", fontdict=font1)
    plt.xlabel("Generation", fontdict=font2)
    plt.ylabel("Best Fitness Value",fontdict=font2)
    plt.subplot(2,2,2)
    plt.scatter([i for i in range(1,numberOfGenes+1)], bestChromosome.genes, s=50,  marker="X" )
    plt.grid()
    plt.title("Genes of the best chromosome", fontdict=font1)
    plt.ylabel("Value of the gene", fontdict=font2)
    plt.xlabel("Position of the gene in Chromosome",fontdict=font2)
    plt.show()



population = populationInitialize(2,8)
visualizePopulation(population)
print([c.calculateFitnessValueOfTheChromosome(fitnessFunction) for c in population])
[population[0] , population[1]] = crossover(population[0], population[1], 0.85, 'single')
visualizePopulation(population)
population[0] = mutation(population[0], 0.10)
population[1] = mutation(population[1], 0.10)
visualizePopulation(population)

binaryGeneticAlgorithm(10, 10, 100, 0.85, 'single', 0.10, 0.20,fitnessFunction)