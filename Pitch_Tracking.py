import matplotlib.pyplot as plt
import numpy
from scipy.io.wavfile import read

FILENAME = "oboe_a4.wav" # wav file to test
W = 1024  # integration window size
THRESHOLD = 0.1  # absolute threshold


def main(f0_min=100, f0_max=500):
    # Prompt user to input desired wav file
    # user_input_directory = input("Please enter the wav file to process:\n")
    # Process desired wav file
    # sample_rate, audio_data = load_wav_data_in_float(user_input_directory)

    sample_rate, audio_data = read(FILENAME)

    # Normalize the audio data
    audio_data = audio_data / numpy.amax(audio_data)

    # Plot the input audio data
    plot(audio_data, "Audio Data", "Time (samples)", "Amplitude")

    # Calculation for the range of tau
    tau_min = int(sample_rate / f0_max)
    tau_max = int(sample_rate / f0_min)
    print("tau_min: " + str(tau_min) + "\ntau_max: " + str(tau_max))

    # Set some parameter values
    t = 50000  # look at the audio from time index sample number 50000 onwards

    # Step 1: The auto-correlation (ACF) method
    # In this step, there are 3 correlation equations.
    # Equation 1 is the original correlation equation.
    correlation = equation_1_acf(audio_data, t, tau_max)
    plot(correlation, "Autocorrelation Function", "lag (samples)", "Correlation")

    # Equation 2 and 3 are improvements to help with reducing errors.
    correlation = equation_2_acf(audio_data, t, tau_max)
    plot(correlation, "Autocorrelation Function", "lag (samples)", "Correlation")

    # Step 2: The difference function
    # This step replaces Step 1.
    difference_function = equation_6_difference_function(audio_data, t, tau_max)
    plot(difference_function, "Difference Function", "lag (samples)", "difference")

    # Step 3: The cumulative mean normalized difference function
    cmndf = equation_8_cumulative_mean_normalized_difference_function(difference_function, tau_max)
    plot(cmndf, "Cumulative Mean Normalized Difference Function", "lag (samples)", "difference")

    # Get fundamental period of a window based on the cmndf
    fundamental_period = detect_pitch(cmndf, tau_min, tau_max, THRESHOLD)
    print("fundamental_period: " + str(fundamental_period))

    pitch = 0
    # If a pitch is detected
    if fundamental_period != 0:
        pitch = float(sample_rate / fundamental_period)
        harmonic_rate = cmndf[int(fundamental_period)]
    else:
        harmonic_rate = min(cmndf)

    print("pitch: " + str(pitch))
    print("harmonic_rate: " + str(harmonic_rate))


def plot(data, title, x_label, y_label):
    """
    Utility function to plot data.

    :param data: data to plot
    :param title: title on graph
    :param x_label: label on x axis
    :param y_label: label on y axis
    """
    print(data)
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def equation_1_acf(x, t, tau_max):
    """
    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (1) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: maxmum lag
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max
    for j in range(t + 1, t + W):
        for tau in range(0, tau_max):
            correlation[tau] += x[j] * x[j+tau]

    return correlation


def equation_2_acf(x, t, tau_max):
    """

    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (2) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: maximum lag
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max
    for tau in range(1, tau_max):
        for j in range(t+1, t+W-tau):
            correlation[tau] += x[j] * x[j+tau]

    return correlation


def equation_6_difference_function(x, t, tau_max):
    """
    Compute the difference function of audio data x. This is equation (6) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: maximum lag
    :return: difference function
    :rtype: list
    """
    difference_function = [0] * tau_max
    for tau in range(1, tau_max):
        for j in range(t, t+W):
            tmp = (x[j] - x[j + tau])
            difference_function[tau] += tmp * tmp
    return difference_function


def equation_8_cumulative_mean_normalized_difference_function(difference_function, tau_max):
    """
    Compute the cumulative mean normalized difference function. This is equation (8) in the YIN article.

    :param difference_function: difference function
    :param tau_max: maximum lags
    :return: cumulative mean normalized difference function
    :rtype: list
    """
    cmndf = [0] * tau_max
    cmndf[0] = 1
    for tau in range(1, tau_max):
        accumulated_value = 0
        for j in range(1, tau+1):
            accumulated_value += difference_function[j]
        accumulated_value = (1 / tau) * accumulated_value
        cmndf[tau] = difference_function[tau] / accumulated_value
    return cmndf


def detect_pitch(cmndf, tau_min, tau_max, threshold):
    """
    Compute the fundamental period of a window based on the cumulative mean normalized difference function

    :param cmndf: cumulative mean normalized difference function
    :param tau_min: minimum period
    :param tau_max: maximum period
    :param threshold: harmonicity threshold to determine if it is necessary to compute pitch frequency
    :return: fundamental period if there is values under threshold, else 0
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0  # if no pitch detected


main()
