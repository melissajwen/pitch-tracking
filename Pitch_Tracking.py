import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

FILENAME = "four_prosody_2_mono.wav"  # wav file to test
W = 1024  # integration window size
W_STEP = 512  # window step length
F0_MIN = 50  # lower bound of human voiced speech f0
F0_MAX = 500  # upper bound of human voiced speech f0
THRESHOLD = 0.1  # absolute threshold
SMOOTHING_MARGIN = F0_MIN * 0.20  # +20% range from the lower bound f0


def main():
    # Get data and sample rate from file
    sample_rate, audio_data = read(FILENAME)

    # Normalize the audio data
    audio_data = audio_data / np.amax(audio_data)

    # Plot the input audio data
    plot(audio_data, "Audio Data", "Time (samples)", "Amplitude")

    # Establish the beginning time index to remove noise (used only for Autocorrelation Function)
    t = 0

    # Calculation for the range of tau
    tau_min = int(sample_rate / F0_MAX)
    tau_max = int(sample_rate / F0_MIN)

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
    time_scale = range(0, signal_length - W, W_STEP)
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
        # plot(difference_function, "Difference Function", "lag (samples)", "difference")

        # Step 3: The cumulative mean normalized difference function (cmndf)
        cmndf = equation_8_cumulative_mean_normalized_difference_function(difference_function, W, tau_max)
        # plot(cmndf, "Cumulative Mean Normalized Difference Function", "lag (samples)", "difference")

        # Step 4: Absolute threshold
        fundamental_period = absolute_threshold(cmndf, THRESHOLD, tau_min, tau_max)

        better_fundamental_period = 0
        if fundamental_period != 0:
            # Step 5: Parabolic interpolation
            better_fundamental_period = parabolic_interpolation(cmndf, fundamental_period)

            # Step 6: Best local estimate
            best_local_period = best_local_estimate(cmndf, fundamental_period)

        # Gather results
        if np.argmin(cmndf) > tau_min:
            argmins[i] = float(sample_rate / np.argmin(cmndf))
        # If a pitch is found...
        if fundamental_period != 0:
            pitches[i] = float(sample_rate / best_local_period)
            harmonic_rates[i] = cmndf[best_local_period]
        else:
            # Return the global minimum if the threshold is too low to detect period
            harmonic_rates[i] = cmndf.index(min(cmndf))

    plot(pitches, "Pitch Contour", "Window number", "Pitch (Hz)")
    smoothed_contour = smooth_pitch_contour(pitches)
    plot(smoothed_contour, "Smoothed Pitch Contour", "Window number", "Pitch (Hz)")


def smooth_pitch_contour(pitches):
    """
    Compute a smoothed pitch contour through noise omission and average values

    :param pitches: The pitch contour
    :return: Smoothed pitch contour
    """
    length_of_pitches = len(pitches)
    smoothed_pitches = [0.0] * length_of_pitches
    for i in range(0, length_of_pitches):
        if pitches[i] > F0_MIN + SMOOTHING_MARGIN:
            smoothed_pitches[i] = pitches[i]

    # If there is a short burst in frequency, classify it as noise and cancel it
    for i in range(0, length_of_pitches):
        count = 0
        if smoothed_pitches[i] != 0:
            while smoothed_pitches[i+count] != 0:
                count += 1

            if count < 6:
                for j in range(0, count):
                    smoothed_pitches[i+j] = 0.0

    for i in range(0, length_of_pitches - 1):
        # If there is a big drop of about 35Hz, and it is not due to it
        # being a value of 0, we even out the contour
        if smoothed_pitches[i] - smoothed_pitches[i+1] >= 35:
            if smoothed_pitches[i+1] != 0:
                smoothed_pitches[i+1] = smoothed_pitches[i]

    # Smooth out the pitch contour by assigning the average of the next 6 pitch points
    range_for_smoothing = 6
    for i in range(0,  length_of_pitches - range_for_smoothing):
        if smoothed_pitches[i] != 0:
            average_value_of_samples = 0
            count = 0
            for j in range(0, range_for_smoothing):
                if smoothed_pitches[i+j] != 0:
                    average_value_of_samples += smoothed_pitches[i+j]
                    count += 1
            if count > 0:
                average_value_of_samples /= count

                for j in range(0, count):
                    smoothed_pitches[i+j] = average_value_of_samples

    return smoothed_pitches


def plot(data, title, x_label, y_label):
    """
    Utility function to plot data.

    :param data: data to plot
    :param title: title on graph
    :param x_label: label on x axis
    :param y_label: label on y axis
    """
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


def parabolic_interpolation(cmndf, fundamental_period):
    """
    Compute a better fundamental period based off the previous estimated one

    :param cmndf: cumulative mean normalized difference function
    :param fundamental_period: the estimated fundamental period
    :return: same or better fundamental period
    """
    if fundamental_period < 1:
        x0 = fundamental_period
    else:
        x0 = fundamental_period - 1

    if fundamental_period + 1 < len(cmndf):
        x2 = fundamental_period + 1
    else:
        x2 = fundamental_period

    if x0 == fundamental_period:
        if cmndf[fundamental_period] <= cmndf[x2]:
            return fundamental_period
        else:
            return x2

    if x2 == fundamental_period:
        if cmndf[fundamental_period] <= cmndf[x0]:
            return fundamental_period
        else:
            return x0

    s0 = cmndf[x0]
    s1 = cmndf[fundamental_period]
    s2 = cmndf[x2]
    return int(fundamental_period + 0.5 * (s2 - s0) / (2 * s1 - s2 - s0))


def best_local_estimate(cmndf, fundamental_period):
    """
    Compute the best local estimate

    :param cmndf: cumulative mean normalized difference function
    :param fundamental_period: the estimated fundamental period
    :return: same or better fundamental period
    """
    # Search range must be within 0.0 ~ 1.0
    search_range = 0.2
    i = fundamental_period + 1
    n = len(cmndf)
    k = cmndf[fundamental_period]
    initial_k = cmndf[fundamental_period]
    smallest_value = fundamental_period

    while i < n:
        if (fundamental_period / i) > search_range:
            break
        if cmndf[i] < k:
            k = cmndf[i]
            smallest_value = i
            break
        i += 1

    if k == initial_k:
        i = fundamental_period - 1
        while i > 0:
            if (i / fundamental_period) < (1 - search_range):
                break
            if cmndf[i] < k:
                k = cmndf[i]
                smallest_value = i
                break
            i -= 1

    return smallest_value


main()
