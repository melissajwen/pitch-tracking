import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

#FILENAME = "oboe_a4.wav"  # wav file to test
FILENAME = "four_prosody_mandarin_mono.wav"  # wav file to test
W = 1024  # integration window size
W_STEP = 512
F0_MIN = 100  # lower bound of human voiced speech f0
F0_MAX = 500  # upper bound of human voiced speech f0
THRESHOLD = 0.1  # absolute threshold
SMOOTH_THRESHOLD = 30  # The threshold was determined by experiments to be 30Hz


def main():
    # Get data and sample rate from file
    sample_rate, audio_data = read(FILENAME)

    # Normalize the audio data
    audio_data = audio_data / np.amax(audio_data)

    # Plot the input audio data
    plot(audio_data, "Audio Data", "Time (samples)", "Amplitude")

    # Establish the beginning time index to remove noise
    t = 50000

    # Calculation for the range of tau
    tau_min = int(sample_rate / F0_MAX)
    tau_max = int(sample_rate / F0_MIN)
    # print("tau_min: " + str(tau_min) + "\n" + "tau_max: " + str(tau_max))

    # Step 1: The auto-correlation (ACF) method
    # Equation (1) is the original correlation equation
    correlation = equation_1_acf(audio_data, W, t, tau_max)
    plot(correlation, "Autocorrelation Function", "lag (samples)", "Correlation")

    # Equation (2) is an improvement on (1) to reduce errors
    correlation = equation_2_acf(audio_data, W, t, tau_max)
    plot(correlation, "Autocorrelation Function", "lag (samples)", "Correlation")

    # Set up for f0 contour
    signal_length = len(audio_data)
    # Time values for each analysis window
    time_scale = range(0, signal_length - W, W_STEP)  # worked (to be removed)
    times = [t/float(sample_rate) for t in time_scale]  # worked (to be removed)
    # Split up the signal into multiple window frames
    frames = [audio_data[t:t + W] for t in time_scale]

    time_scale_length = len(time_scale)
    pitches = [0.0] * time_scale_length
    # harmonic_rates is a list of harmonic rate values for each fundamental frequency value
    harmonic_rates = [0.0] * time_scale_length
    # argmins stores the minimums of the cmndf
    argmins = [0.0] * time_scale_length

    for i, frame in enumerate(frames):

        # YIN computation
        # Step 2: The difference function
        difference_function = equation_6_difference_function_fastest(frame, t, tau_max)
        #plot(difference_function, "Difference Function", "lag (samples)", "difference")

        # Step 3: The cumulative mean normalized difference function (cmndf)
        cmndf = equation_8_cumulative_mean_normalized_difference_function(difference_function, W, tau_max)
        #plot(cmndf, "Cumulative Mean Normalized Difference Function", "lag (samples)", "difference")

        # Step 4: Absolute threshold
        fundamental_period = absolute_threshold(cmndf, THRESHOLD, tau_min, tau_max)
        #print("fundamental_period: " + str(fundamental_period))

        # Gather results
        if np.argmin(cmndf) > tau_min:
            argmins[i] = float(sample_rate / np.argmin(cmndf))
        # If a pitch is found...
        if fundamental_period != 0:
            pitches[i] = float(sample_rate / fundamental_period)
            harmonic_rates[i] = cmndf[fundamental_period]
        else:
            # Return the global minimum if the threshold is too low to detect period
            harmonic_rates[i] = cmndf.index(min(cmndf))

    plot(pitches, "Pitch Contour", "Window number", "Pitch (Hz)")


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


def equation_1_acf(x, W, t, tau_max):
    """
    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (1) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: integration window size
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max
    for j in range(t + 1, t + W):
        for tau in range(0, tau_max):
            correlation[tau] += x[j] * x[j + tau]

    return correlation


def equation_2_acf(x, W, t, tau_max):
    """

    Compute the correlation between an audio signal "x" and its shifted self of lag "tau" value.
    This is equation (2) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: integration window size
    :return: autocorrelation function
    :rtype: float
    """
    correlation = [0] * tau_max

    for tau in range(0, tau_max):
        for j in range(t + 1, t + W - tau):
            correlation[tau] += x[j] * x[j + tau]

    return correlation


def equation_6_difference_function_original(x, W, t, tau_max):
    """
    Compute the difference function of audio data x. This is equation (6) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """
    difference_function = [0] * tau_max
    for tau in range(0, tau_max):
        for j in range(1, t + W):
            tmp = (x[j] - x[j + tau])
            difference_function[tau] += tmp * tmp
    return difference_function


def equation_6_difference_function_fastest(x, t, tau_max):
    """
    Compute the difference function of audio data x. This is equation (6) in the YIN article.

    :param x: audio data
    :param t: time index
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def equation_8_cumulative_mean_normalized_difference_function(difference_function, W, tau_max):
    """
    Compute the cumulative mean normalized difference function. This is equation (8) in the YIN article.

    :param difference_function: difference function from Step 2
    :param tau_max: integration window size
    :return: cumulative mean normalized difference function
    :rtype: list
    """
    cmndf = [0] * tau_max
    cmndf[0] = 1.0
    for tau in range(1, tau_max):
        # Sum up all values in difference_function
        accumulated_value = 0
        for j in range(1, tau + 1):
            accumulated_value += difference_function[j]
        accumulated_value = (1 / tau) * accumulated_value
        cmndf[tau] = difference_function[tau] / accumulated_value
    return cmndf


def absolute_threshold(cmndf, threshold, tau_min, tau_max):
    """
    Compute the fundamental period given the cumulative mean normalized difference function.

    :param cmndf: cumulative mean normalized difference function
    :param tau_min: minimum period for speech
    :param tau_max: maximum period for speech
    :param threshold: threshold to determine if it is necessary to compute pitch frequency
    :return: fundamental period if there is values under threshold. if there aren't any, return global minimum of cmndf
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmndf[tau] < threshold:
            # Find the minimum below this threshold, not just the first value below the threshold
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1

            return tau
        tau += 1

    # If no pitch is detected...
    return 0


main()
