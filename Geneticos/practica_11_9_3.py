import os
import numpy
import imageio
import pygad
import matplotlib.pyplot
import numpy
import functools
import operator
from keras.preprocessing.image import load_img


def img2chromosome(img_arr):
    """
    Represents the image as a 1D vector.

    img_arr: The image to be converted into a vector.

    Returns the vector.
    """

    return numpy.reshape(img_arr, (functools.reduce(operator.mul, img_arr.shape)))
import numpy as np
import functools
import operator
import os

def chromosome2img(vector, shape):
    """
    Converts a 1D vector into an array with the given shape.

    vector: The vector to be converted into an array.
    shape: The shape of the target array.

    Returns the array.
    """

    # Check if the vector can be reshaped according to the specified shape.
    if len(vector) != functools.reduce(operator.mul, shape, 1):
        raise ValueError(f"A vector of length {len(vector)} cannot be reshaped into an array of shape {shape}.")

    return np.reshape(vector,shape)

def fitness_fun(solution, solution_idx):
    """
    Calculating the fitness value for a solution in the population.
    The fitness value is calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes.

    solution: Current solution in the population to calculate its fitness.
    solution_idx: Index of the solution within the population.
    """

    fitness = numpy.sum(numpy.abs(target_chromosome-solution))

    # Negating the fitness value to make it increasing rather than decreasing.
    fitness = numpy.sum(target_chromosome) - fitness
    return fitness
def callback(ga_instance):
    
    best_chromosome, best_fitness ,_= ga_instance.best_solution()
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {best_fitness}")

    output_dir = 'solutions'
    os.makedirs(output_dir, exist_ok=True)
    if ga_instance.generations_completed % 5000 == 0:
        matplotlib.pyplot.imsave(
            os.path.join(output_dir, f'solution_{ga_instance.generations_completed}.png'),
            chromosome2img(best_chromosome, target_im.shape)
        )

    
        

from PIL import Image
import requests
from io import BytesIO

response = requests.get('https://github.com/M-Yerro/IA2025/blob/main/07-IA2025%20fruit.jpg?raw=true')
# Open the image form working directory
image = Image.open(BytesIO(response.content))
print(image.format)
print(image.size)
print(image.mode)
# show the image
image.show()
print(image)
target_im = numpy.asarray(image)
target_im = numpy.asarray(target_im/255, dtype=float)

# Target image after enconding. Value encoding is used.
target_chromosome = img2chromosome(target_im)
print(target_im)
ga_instance = pygad.GA(num_generations=260000,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=20,
                       num_genes=target_im.size,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=0.01,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       callback_generation=callback)

ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

result = chromosome2img(solution, target_im.shape)

matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("PyGAD & GARI for Reproducing Images")
matplotlib.pyplot.show()
