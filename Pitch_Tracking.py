import matplotlib.pyplot as plt
import numpy
from scipy.io.wavfile import read


def main(f0_min = 100, f0_max = 500):
    # Prompt user to input desired wav file
    user_input_directory = input("Please enter the wav file to process:\n")
    # Process desired wav file
    sample_rate, audio_data = load_wav_data_in_float(user_input_directory)
    # Normalize the audio data
    audio_data = audio_data / numpy.amax(audio_data)

    # Plot the input audio data
    plt.plot(audio_data)
    plt.title("Audio Data")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

    # Calculation for the range of tau
    tau_min = int(sample_rate / f0_max)
    tau_max = int(sample_rate / f0_min)
    print("tau_min: " + str(tau_min))
    print("tau_max: " + str(tau_max))

    # Set some parameter values
    t = 50000  # look at the audio from time index sample number 50000 onwards
    W = 1024  # integration window size

    # Step 1: The auto-correlation (ACF) method
    # In this step, there are 3 correlation equations.
    # Equation 1 is the original correlation equation.
    correlation = equation_1_acf(audio_data, W, t, tau_max)
    print(correlation)
    plt.plot(correlation)
    plt.title("Autocorrelation Function")
    plt.xlabel("lag (samples)")
    plt.ylabel("Correlation")
    plt.show()
    # Equation 2 and 3 are improvements to help with reducing errors.
    correlation = equation_2_acf(audio_data, W, t, tau_max)
    print(correlation)
    plt.plot(correlation)
    plt.title("Autocorrelation Function")
    plt.xlabel("lag (samples)")
    plt.ylabel("Correlation")
    plt.show()

    # Step 2: The difference function
    # In this step,
    difference_function = equation_6_difference_function(audio_data, W, t, tau_max)
    print(difference_function)
    plt.plot(difference_function)
    plt.title("Difference Function")
    plt.xlabel("lag (samples)")
    plt.ylabel("difference")
    plt.show()

    # Step 3: The cumulative mean normalized difference function
    cmndf = equation_8_cumulative_mean_normalized_difference_function(difference_function, W, tau_max)
    print(cmndf)
    plt.plot(cmndf)
    plt.title("Cumulative Mean Normalized Difference Function")
    plt.xlabel("lag (samples)")
    plt.ylabel("difference")
    plt.show()


def equation_1_acf(x, W, t, tau_max):
    """

    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (1) in the YIN article.

    :param x: audio data
    :param W: integration window size
    :param t: time index
    :param tau: lag
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max
    for j in range(t + 1, t + W):
        for tau in range(0, tau_max):
            correlation[tau] += x[j] * x[j+tau]

    return correlation


def equation_2_acf(x, W, t, tau_max):
    """

    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (2) in the YIN article.

    :param x: audio data
    :param W: integration window size
    :param t: time index
    :param tau: lag
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max
    for tau in range(1, tau_max):
        for j in range(t+1, t+W-tau):
            correlation[tau] += x[j] * x[j+tau]

    return correlation


def equation_6_difference_function(x, W, t, tau_max):
    """
    Compute the difference function of audio data x. This is equation (6) in the YIN article.

    :param x: audio data
    :param W: window length
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """
    difference_function = [0] * tau_max
    for tau in range(1, tau_max):
        for j in range(t, t+W):
            tmp = (x[j] - x[j + tau])
            difference_function[tau] += tmp * tmp
    return difference_function


def equation_8_cumulative_mean_normalized_difference_function(difference_function, W, tau_max):
    """
    Compute the cumulative mean normalized difference function. This is equation (8) in the YIN article.

    :param difference_function: difference function
    :param W: window length
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = [0] * tau_max
    for tau in range(0, tau_max):
        if tau == 0:
            cmndf[tau] = 1
        else:
            for j in range(1, tau):
                accumulated_value = numpy.cumsum(difference_function[j:tau])[0]
                cmndf[tau] = difference_function[tau] / ((1 / tau) * accumulated_value)
    return cmndf


def load_wav_data_in_float(audio_file_directory):
    wav_file_information = read(audio_file_directory)
    audio_data = wav_file_information[1]
    sample_rate = wav_file_information[0]
    return sample_rate, audio_data


main()
