"""
This is a helper module to interpolate (and extrapolate) between a set of given latent vectors.

"""

import numpy as np


def latentVectorInterpolate(input_vectors, output_count=3, method="linear", rank=None):
    """
    Function to fit a polynomial to given latent vectors and generate new latent vectors between the given one.
    :param rank: Rank of custom polynomial to fit
    :param input_vectors: list of 2 or more given latent vectors
    :param output_count: total number of desired output latent vectors (including the input vectors)
    :param method: interpolation method
    :return: list of given and interpolated latent vectors
    """

    assert len(input_vectors) > 0, "Please supply input vectors"

    if method == "linear" and len(input_vectors) == 2:
        steps = np.linspace(0, 1, num=output_count)
        output_vectors = list()
        for step in steps:
            new_vector = (1.0 - step) * input_vectors[0] + step * input_vectors[1]
            output_vectors.append(new_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "linear":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 2))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 1)
        output_vectors = list()
        for step in np.linspace(0, len(input_vectors)-1, num=output_count):
            current_vector = coefficients[:, 0] * step + coefficients[:, 1]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "quadratic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 3))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 2)
        output_vectors = list()
        for step in np.linspace(0, len(input_vectors)-1, num=output_count):
            current_vector = coefficients[:, 0] * step**2 + coefficients[:, 1] * step + coefficients[:, 2]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "cubic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 4))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 3)
        output_vectors = list()
        for step in np.linspace(0, len(input_vectors)-1, num=output_count):
            current_vector = (coefficients[:, 0] * step ** 3 + coefficients[:, 1] * step ** 2
                              + coefficients[:, 2] * step + coefficients[:, 3])
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "customrank":
        assert rank is not None, "Please specify a rank for the polynomials to fit"
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], rank+1))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension]
                                                        for latent_vector in input_vectors], rank)
        output_vectors = list()
        for step in np.linspace(0, len(input_vectors) - 1, num=output_count):
            current_vector = coefficients[:, 0] * step ** rank
            for i in range(1, rank+1):
                current_vector += coefficients[:, i] * step ** (rank-i)
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    else:
        print("Please specify a valid interpolation method. Choices are: linear, quadratic, cubic, customrank")
        raise NotImplementedError


def latentVectorExtrapolateForward(input_vectors, output_count=3, method="linear", step_ratio=1):
    """
    Function to fit a polynomial to given latent vectors and generate new latent vectors later in
    time than the given ones.
    :param input_vectors: list of 2 or more given latent vectors
    :param output_count: total number of desired output latent vectors (including the input vectors)
    :param method: extrapolation method
    :param step_ratio: distance between extrapolated vectors normalized by the distance between the input vectors
    :return: list of given and extrapolated latent vectors
    """
    assert len(input_vectors) > 0, "Please supply input vectors"

    if method == "linear" and len(input_vectors) == 2:
        output_vectors = list()
        output_vectors.append(input_vectors[0])
        for step in range(0, output_count-1):
            new_vector = input_vectors[1] + step_ratio * step * (input_vectors[1] - input_vectors[0])
            output_vectors.append(new_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "linear":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 2))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 1)
        output_vectors = list()
        for step in np.linspace(0, output_count-1, num=output_count):
            if step >= len(input_vectors):
                current_vector = (coefficients[:, 0] * (len(input_vectors) - 1
                                  + (step - len(input_vectors) + 1) * step_ratio)
                                  + coefficients[:, 1])
            else:
                current_vector = coefficients[:, 0] * step + coefficients[:, 1]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "quadratic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 3))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 2)
        output_vectors = list()
        for step in np.linspace(0, output_count - 1, num=output_count):
            if step >= len(input_vectors):
                current_vector = (coefficients[:, 0] * (len(input_vectors) - 1
                                  + (step - len(input_vectors) + 1) * step_ratio) ** 2 + coefficients[:, 1]
                                  * (len(input_vectors) - 1 + (step - len(input_vectors) + 1)
                                  * step_ratio) + coefficients[:, 2])
            else:
                current_vector = coefficients[:, 0] * step**2 + coefficients[:, 1] * step + coefficients[:, 2]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "cubic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 4))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 3)
        output_vectors = list()
        for step in np.linspace(0, output_count - 1, num=output_count):
            if step >= len(input_vectors):
                current_vector = (coefficients[:, 0] * (len(input_vectors) - 1 + (step - len(input_vectors) + 1)
                                  * step_ratio) ** 3 + coefficients[:, 1] * (len(input_vectors) - 1
                                  + (step - len(input_vectors) + 1) * step_ratio) ** 2 + coefficients[:, 2]
                                  * (len(input_vectors) - 1 + (step - len(input_vectors) + 1)
                                  * step_ratio) + coefficients[:, 3])
            else:
                current_vector = (coefficients[:, 0] * step**3 + coefficients[:, 1] * step**2 + coefficients[:, 2]
                                  * step + coefficients[:, 3])
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    else:
        print("Please specify a valid interpolation method. Choices are: linear, quadratic, cubic")
        raise NotImplementedError


def latentVectorExtrapolateBackward(input_vectors, output_count=3, method="linear", step_ratio=1):
    """
    Function to fit a polynomial to given latent vectors and generate new latent vectors between the given
    one earlier in time than the given ones.
    :param input_vectors: list of 2 or more given latent vectors
    :param output_count: total number of desired output latent vectors (including the input vectors)
    :param method: extrapolation method
    :param step_ratio: distance between extrapolated vectors normalized by the distance between the input vectors
    :return: list of given and extrapolated latent vectors
    """
    assert len(input_vectors) > 0, "Please supply input vectors"

    if method == "linear" and len(input_vectors) == 2:
        output_vectors = list()
        for step in range(output_count-2, -1, -1):
            new_vector = input_vectors[0] - step_ratio * step * (input_vectors[1] - input_vectors[0])
            output_vectors.append(new_vector)
        output_vectors.append(input_vectors[1])
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "linear":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 2))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 1)
        output_vectors = list()
        for step in np.linspace(- output_count + len(input_vectors), len(input_vectors)-1, num=output_count):
            if step >= len(input_vectors):
                current_vector = coefficients[:, 0] * (
                            len(input_vectors) - 1 + (step - len(input_vectors) + 1) * step_ratio) + coefficients[:, 1]
            else:
                current_vector = coefficients[:, 0] * step + coefficients[:, 1]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "quadratic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 3))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 2)
        output_vectors = list()
        for step in np.linspace(- output_count + len(input_vectors), len(input_vectors) - 1, num=output_count):
            if step >= len(input_vectors):
                current_vector = coefficients[:, 0] * (
                            len(input_vectors) - 1 + (step - len(input_vectors) + 1) * step_ratio) ** 2 + coefficients[
                                                                                                          :, 1] * (
                                             len(input_vectors) - 1 + (
                                                 step - len(input_vectors) + 1) * step_ratio) + coefficients[:, 2]
            else:
                current_vector = coefficients[:, 0] * step ** 2 + coefficients[:, 1] * step + coefficients[:, 2]
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    elif method == "cubic":
        t = np.arange(len(input_vectors))
        coefficients = np.zeros((input_vectors[0].shape[0], 4))
        for dimension in range(input_vectors[0].shape[0]):
            coefficients[dimension, :] = np.polyfit(t, [latent_vector[dimension] for latent_vector in input_vectors], 3)
        output_vectors = list()
        for step in np.linspace(- output_count + len(input_vectors), len(input_vectors) - 1, num=output_count):
            if step >= len(input_vectors):
                current_vector = (coefficients[:, 0] * (len(input_vectors) - 1 + (step - len(input_vectors) + 1)
                                  * step_ratio) ** 3 + coefficients[:, 1] * (len(input_vectors) - 1
                                  + (step - len(input_vectors) + 1) * step_ratio) ** 2 + coefficients[:, 2]
                                  * (len(input_vectors) - 1 + (step - len(input_vectors) + 1)
                                  * step_ratio) + coefficients[:, 3])
            else:
                current_vector = (coefficients[:, 0] * step ** 3 + coefficients[:, 1] * step ** 2
                                  + coefficients[:, 2] * step + coefficients[:, 3])
            output_vectors.append(current_vector)
        assert len(output_vectors) == output_count

        return output_vectors

    else:
        print("Please specify a valid interpolation method. Choices are: linear, quadratic, cubic")
        raise NotImplementedError
