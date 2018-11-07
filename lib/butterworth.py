import numpy as np

# An ordering on complex that lets real roots dominate.
#
def sort_complex(x, y):
    if x.real < y.real:
        return True
    elif y.real < x.real:
        return False
    return x.imag < y.imag

def has_positive_imag(z):
    return z.imag > 0

def has_negative_imag(z):
    return z.imag < 0

# Return the polynomial coefficients for roots.
# Keep roots sorted with sortComplex() to maintain precision.
# <b>param roots</b> - numpy array of complex numbers
def get_polynomial_coefficients(roots):

    coeffs = roots.tolist()
    coeffs.sort(key=lambda coord: sort_complex)
    coeffs = [1.0] + coeffs
    sofar = 1
    for k in range(0, len(roots)):
        w = -roots[k]
        j = sofar
        while j > 0:
            coeff_list[j] = coeffs[j] * w + coeffs[j-1]
            j -= 1
        coeffs[0] += w
        k += 1
        sofar += 1

    result = coeffs

    pos_roots = roots
    pos_roots = [root for root in roots if not has_negative_imag(root)]
    pos_roots.tolist().sort(key=lambda coord: sort_complex)

    neg_roots = roots
    neg_roots = [root for root in roots if not has_positive_imag(root)]
    neg_roots.tolist().sort(key=lambda coord: sort_complex)

    same = len(neg_roots) == len(pos_roots) and (pos_roots == neg_roots).all()
    if same:
        for k in range(0, len(coeffs)):
            result[k] = coeffs[k]
            k += 1
    return result


# Write into a and b the real polynomial transfer function coefficients
# from gain, zeros, and poles.
# params: all params except gain are numpy arrays
#
def zeros_poles_to_transfer_coefficients(zeros, poles, gain, a, b):
    a = get_polynomial_coefficients(poles)
    b = get_polynomial_coefficients(zeros)
    for k in range (0, len(b)):
        b[k] *= gain
        k += 1

# Normalize the polynomial representation of the real transfer
# coefficients in a and b.
#
# Output coefficients will also be real numbers.
# Remove leading zeros to avoid dividing by 0.
# params: all params are lists
#
def normalize(b, a):
    while a[0] == 0.0 and len(a) > 1:
        a.pop(0)
    leading_coeff = a[0]
    for k in range(0, len(a)):
        a[k] /= leading_coeff
    for k in range(0, len(b)):
        b[k] /= leading_coeff


# Return the binomial coefficient: n choose k.
#
def choose(n, k):
    if k > n:
        return 0
    if k * 2 > n:
        k = n - k;
    if k == 0:
        return 1
    result = n
    for i in range(2; k-1):
        result *= (n - i + 1)
        result /= i
        i += 1
    return result


# Use the bilinear transform to convert the analog filter coefficients in
# a and b into a digital filter for the sampling frequency fs (1/T).
#
def bilinear_transform(b, a, fs):
    D = len(a) - 1
    N = len(b) - 1
    M = max(N, D)
    Np = M
    Dp = M

    std::vector< std::complex<double> > bprime(Np + 1, 0.0);
    for (unsigned j = 0; j < Np + 1; ++j) {
        std::complex<double> val = 0.0;
        for (unsigned i = 0; i < N + 1; ++i) {
            for (unsigned k = 0; k < i + 1; ++k) {
                for (unsigned l = 0; l < M - i + 1; ++l) {
                    if (k + l == j) {
                        val += std::complex<double>(choose(i, k))
                            *  std::complex<double>(choose(M - i, l))
                            *  b[N - i] * pow(2.0 * fs, i) * pow(-1.0, k);
                    }
                }
            }
        }
        bprime[j] = real(val);
    }

    std::vector< std::complex<double> > aprime(Dp + 1, 0.0);
    for (unsigned j = 0; j < Dp + 1; ++j) {
        std::complex<double> val = 0.0;
        for (unsigned i = 0; i < D + 1; ++i) {
            for(unsigned k = 0; k < i + 1; ++k) {
                for(unsigned l = 0; l < M - i + 1; ++l) {
                    if (k + l == j) {
                        val += std::complex<double>(choose(i, k))
                            *  std::complex<double>(choose(M - i, l))
                            *  a[D - i] * pow(2.0 * fs, i) * pow(-1.0, k);
                    }
                }
            }
        }
        aprime[j] = real(val);
    }

    normalize(bprime, aprime);
    a = aprime;
    b = bprime;
}


// Transform a and b coefficients of transfer function
// into a low-pass filter with cutoff frequency w_0.
// Assume the transfer function has only real coefficients.
//
static void toLowpass(std::vector< std::complex<double> > &b,
                      std::vector< std::complex<double> > &a,
                      double w0)
{
    std::vector<double> pwo;
    const int d = a.size();
    const int n = b.size();
    const int M = int(std::max(double(d), double(n)));
    const unsigned int start1 = int(std::max(double(n - d), 0.0));
    const unsigned int start2 = int(std::max(double(d - n), 0.0));
    for (int k = M - 1; k > -1; --k) pwo.push_back(pow(w0, double(k)));
    unsigned int k;
    for (k = start2; k < pwo.size() && k - start2 < b.size(); ++k) {
        b[k - start2]
            *= std::complex<double>(pwo[start1])
            /  std::complex<double>(pwo[k]);
    }

    for (k = start1; k < pwo.size() && k - start1 < a.size(); ++k) {
        a[k - start1]
            *= std::complex<double>(pwo[start1])
            /  std::complex<double>(pwo[k]);
    }
    normalize(b, a);
}


// Compute zeros, poles and gain for filter of order N assuming the
// normalized Butterworth form of transfer function.
//
// The gain is always 1.0, but parameterized to agree with textbooks.
//
static void
prototypeAnalogButterworth(unsigned N,
                           std::vector< std::complex<double> > &zeros,
                           std::vector< std::complex<double> > &poles,
                           double &gain)
{
    static const std::complex<double> j = std::complex<double>(0, 1.0);
    for (unsigned k = 1; k < N + 1; ++k) {
        poles.push_back(exp(j * (2.0 * k - 1) / (2.0 * N) * M_PI) * j);
    }
    gain = 1.0;
    zeros.clear();
}


// Tangentially warp the Wn input analog frequency to W0
// for the resulting digital filter.
//
// W0 is the warped (high or low) bandpass cutoff frequency.
//
// See http://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l6.pdf
//
void butterworth(unsigned int N, double Wn,
                 std::vector<double> &out_a,
                 std::vector<double> &out_b)
{
    static const double fs = 2.0;
    const double w0 = 2.0 * fs * tan(M_PI * Wn / fs);
    std::vector< std::complex<double> > zeros, poles;
    double gain;
    prototypeAnalogButterworth(N, zeros, poles, gain);
    std::vector< std::complex<double> > a, b;
    zerosPolesToTransferCoefficients(zeros, poles, gain, a, b);
    toLowpass(b, a, w0);
    bilinearTransform(b, a, fs);
    out_a.clear();
    for (unsigned k = 0; k < a.size(); ++k) out_a.push_back(std::real(a[k]));
    out_b.clear();
    for (unsigned k = 0; k < b.size(); ++k) out_b.push_back(std::real(b[k]));
}
