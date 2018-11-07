import numpy as np
import scipy
import time
import cv2
import pylab
import os
import sys
import eulerian_magnification as em
import threading as th
import time as t

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 300 # 30 fps for 10 seconds
        self.hamming_window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])

        self.find_faces = True
        # time gap to wait before bpm can be measured
        self.gap = 0
        # thread for doing evm in background
        self.evmThread = []

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift


    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def getForehead(self):
        return self.get_subface_coord(0.5, 0.18, 0.25, 0.15)

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]

    	# Oxydized hemoglobin (HbO2) absorbs more green light and reflects more red.
        # Deoxydized hemoglobin (HbCO) absorbs more red light and reflects more green.
        # At the moment of pulsation, saturation of HbO2 is maximal and saturation of HbCO is minimal.
        # Note that HbCO contribution to absorbance can be neglected,
        # because it is present in a smaller concentration (>94% of HbO2 and <2% of HbCO).
        # More noise will be present at shorter wavelengths, particularly in blue channel:
        #   - The ability of light to pass through skin decreases with shorter wavelengths.
        #   - Absorption coefficients vary a lot for blue channel:
        #       there is a distinctive peak at 425nm with e=130cm2/mol, and there are values as low as 6cm2/mol at 490nm,
        #       which makes it hard to determine mean absorption rate for this channel
    	# See B.L.Horecker [http://www.jbc.org/content/148/1/173.full.pdf] and similar studies for more details.

        # The coefficients below are estimated empirically, by examining results of hemoglobin spectrography studies
        HBO2_MEAN_ABSORBPTION_RATE_GREEN    = 0.82
        HBO2_MEAN_ABSORBPTION_RATE_RED      = 0.3

        b = np.mean(subframe[:, :, 0])
        g = np.mean(subframe[:, :, 1]) * HBO2_MEAN_ABSORBPTION_RATE_GREEN
        r = np.mean(subframe[:, :, 2]) * HBO2_MEAN_ABSORBPTION_RATE_RED

        return (g + r)

    def drawText(self, text, pos, scale=1.25):
        textColor = (100, 255, 100)
        cv2.putText(self.frame_out, text, pos, cv2.FONT_HERSHEY_PLAIN, scale, textColor)
        return

    def drawMenu(self, cameraStr):
        pos = [10, 25] # starting menu text coordinates
        lineHeight = 25
        menuText = ("Press 'C' to change camera (current: %s)" % cameraStr,
                    "Press 'S' to lock face and begin",
                    "Press 'D' to toggle data plot",
                    "Press 'Esc' to quit")
        if self.find_faces: # enable data plot only when face is detected
            menuText = menuText[0:1] + menuText[3:]
        for text in menuText:
            self.drawText(text, tuple(pos))
            pos[1] += lineHeight
        return

    def drawFaceRect(self):
        x,y,w,h = self.getForehead()
        self.draw_rect([x,y,w,h], (0, 255, 0))

        if self.find_faces:
            self.drawText("Forehead", (x, y), 1.5)
            self.draw_rect(self.face_rect, (255, 0, 0))
            x, y, w, h = self.face_rect
            self.drawText("Face", (x, y), 1.5)
        else:
            text = "estimate: %0.0f bpm" % (self.bpm)
            if self.gap:
                text += ", wait %0.0f s" % self.gap
            self.drawText(text, (int(x - w / 2), int(y)), 1)
        return

    def findFaces(self):
        self.data_buffer, self.times = [], []
        detected = list(
                self.face_cascade.detectMultiScale(
                        self.gray,
                        scaleFactor=1.3,
                        minNeighbors=4,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
            )

        if len(detected) > 0:
            # sort found faces, we need only the best match
            detected.sort(key=lambda a: a[-1] * a[-2])
            # set face rectangle to the found face only
            # if it is away from the last detected for more than 10px -
            # this allows some head movement in the frame without triggering
            # face detection too much
            if self.shift(detected[-1]) > 10:
                self.face_rect = detected[-1]

    ''' Adds mean color values from forehead to the buffer.
        Returns True if there is enough data gathered '''
    def gatherColorIntensityData(self):
        #vals = self.get_subface_means(self.getForehead())

        self.data_buffer.append(self.frame_in)
        buffer_len = len(self.data_buffer)
        if buffer_len > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            buffer_len = self.buffer_size
        elif buffer_len < self.buffer_size:
            return False
        return True

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(
                cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY)
            )

        #self.drawMenu(str(cam))
        #self.drawFaceRect()

        #if self.find_faces:
            # face detection not yet done
        #    self.findFaces()
        #    return

        dataGathered = self.gatherColorIntensityData()
        #processed = np.array(self.data_buffer)
        #self.samples = processed
        buffer_len = len(self.data_buffer)

        if len(self.times) > 1:
            self.fps = float(buffer_len) / (self.times[-1] - self.times[0])
            self.gap = (self.buffer_size - buffer_len) / self.fps

        if not dataGathered:
            self.evmThread = th.Thread(
                    target = self.eulerian_magnification,
                    args = (np.array(self.data_buffer), self.fps, 0.98, 1.35, 100, 4, )
                )
            return

        vid = self.load_video_float(np.array(self.data_buffer))
        #em.show_frequencies(vid, fps);

        #self.save_video(vid, self.fps, 'C:\\work\\input.avi')

        if  self.evmThread.is_alive():
            self.evmThread.join()
        self.evmThread = th.Thread(
                target = self.eulerian_magnification,
                args = (vid, self.fps, 0.98, 1.35, 100, 4, )
            )
        self.evmThread.start()

        #todo: count pulse from signal
        #self.save_video(vid, self.fps, 'C:\\work\\output.avi')
        #self.frame_out = vid[0]

    def create_laplacian_video_pyramid(self, video, pyramid_levels):
        return self._create_pyramid(video, pyramid_levels, self.create_laplacian_image_pyramid)

    def _create_pyramid(self, video, pyramid_levels, pyramid_fn):
        vid_pyramid = []
        # frame_count, height, width, colors = video.shape
        for frame_number, frame in enumerate(video):
            frame_pyramid = pyramid_fn(frame, pyramid_levels)

            for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
                if frame_number == 0:
                    vid_pyramid.append(
                        np.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1], 3),
                                    dtype="float"))

                vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

        return vid_pyramid

    def create_gaussian_video_pyramid(self, video, pyramid_levels):
        return self._create_pyramid(video, pyramid_levels, self.create_gaussian_image_pyramid)

    def create_laplacian_image_pyramid(self, image, pyramid_levels):
        gauss_pyramid = self.create_gaussian_image_pyramid(image, pyramid_levels)
        laplacian_pyramid = []
        for i in range(pyramid_levels - 1):
            laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1])) + 0)

        laplacian_pyramid.append(gauss_pyramid[-1])
        return laplacian_pyramid

    def create_gaussian_image_pyramid(self, image, pyramid_levels):
        gauss_copy = np.ndarray(shape=image.shape, dtype="float")
        #print("gauss copy", gauss_copy)
        gauss_copy[:] = image
        img_pyramid = [gauss_copy]
        for pyramid_level in range(1, pyramid_levels):
            gauss_copy = cv2.pyrDown(gauss_copy)
            img_pyramid.append(gauss_copy)

        return img_pyramid

    def temporal_bandpass_filter(self, data, fps, freq_min=0.833, freq_max=1, axis=0, amplification_factor=1):
        print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
        fft = scipy.fftpack.rfft(data, axis=axis)
        frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - freq_min)).argmin()
        bound_high = (np.abs(frequencies - freq_max)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0

        result = np.ndarray(shape=data.shape, dtype='float')
        result[:] = scipy.fftpack.ifft(fft, axis=0)
        result *= amplification_factor
        return result

    def online_riesz_video_magnification(amplification_factor, low_cutoff, high_cutoff, sampling_rate):
        nyquist_frequence = sampling_rate / 2;
        temporal_filter_order = 1;
        #b,a = getButterworthFilterCoefficients(..)
        b,a = self.IIRTemporalFilter

        gaussian_kernel_sd = 2 # 2px
        gaussian_kernel = getGaussianKernel(gaussian_kernel_sd)

        previous_frame = getFirstFrameFromVideo()
        previous_laplacian_pyramid, previous_riesz_x, previous_riedz_y = computeRieszPyramid(previous_frame)
        number_of_levels = numel(previous_laplacian_pyramid) - 1
        for k in range (1, number_of_levels):
            phase_cos[k] = np.zeros(size(previous_laplacian_pyramid[k]))
            phase_sin[k] = np.zeros(size(previous_laplacian_pyramid[k]))
            register0_cos[k] = np.zeros(size(previous_laplacian_pyramid[k]))
            register1_cos[k] = np.zeros(size(previous_laplacian_pyramid[k]))
            register0_sin[k] = np.zeros(size(previous_laplacian_pyramid[k]))
            register1_sin[k] = np.zeros(size(previous_laplacian_pyramid[k]))

        while running:
            current frame = GetNextFrameFromVideo()
            current laplacian pyramid, current riesz x, current riesz y = ComputeRieszPyramid(current frame)

            # We compute a Laplacian pyramid of the motion magnified frame first and then
            # collapse it at the end.
            # The processing in the following loop is processed on each level
            # of the Riesz pyramid independently
            for k in range (1, number_of_levels):
                # Compute quaternionic phase difference between current Riesz pyramid
                # coefficients and previous Riesz pyramid coefficients.
                phase_difference_cos, phase_difference_sin, amplitude = ComputePhaseDifferenceAndAmplitude(
                        current_laplacian_pyramid[k],
                        current_riesz_x[k],
                        current_riesz_y[k],
                        previous_laplacian_pyramid[k],
                        previous_riesz_x[k],
                        previous_riesz_y[k]
                )

                # Adds the quaternionic phase difference to the current value of the quaternionic
                # phase.
                # Computing the current value of the phase in this way is
                # equivalent to phase unwrapping.
                phase_cos[k] = phase_cos[k] + phase_difference_cos
                phase_sin[k] = phase_sin[k] + phase_difference_sin

                # Temporally filter the quaternionic phase using current value and stored
                # information

                phase_filtered_cos, register0_cos[k], register1_cos[k] = IIRTemporalFilter(
                        B, A, phase_cos[k], register0_cos[k], register1_cos[k]
                    )
                phase_filtered_sin, register0_sin[k], register1_sin[k] = IIRTemporalFilter(
                        B, A, phase_sin[k], register0_sin[k], register1_sin[k]
                    )

                # Spatial blur the temporally filtered quaternionic phase signals.
                # This is not an optional step. In addition to denoising,
                # it smooths out errors made during the various approximations.
                phase_filtered_cos = AmplitudeWeightedBlur(phase_filtered_cos, amplitude, gaussian_kernel)
                phase_filtered_sin = AmplitudeWeightedBlur(phase filtered sin, amplitude, gaussian kernel)

                # The motion magnified pyramid is computed by phase shifting
                # the input pyramid by the spatio-temporally filtered quaternionic phase and
                # taking the real part.
                phase_magnified_filtered_cos = amplification_factor * phase_filtered_cos
                phase_magnified_filtered_sin = amplification_factor * phase_filtered_sin

                motion_magnified_laplacian_pyramid[k] = PhaseShiftCoefficientRealPart(
                        current_laplacian_pyramid[k],
                        current_riesz_x[k],
                        current_riesz_y[k],
                        phase_magnified_filtered_cos,
                        phase_magnified_filtered_sin
                    )

            # Take lowpass residual from current frame's lowpass residual
            # and collapse pyramid.
            motion_magnified_laplacian_pyramid[number_of_levels+1] = current_laplacian_pyramid[number_of_levels+1]
            motion_magnified_frame = CollapseLaplacianPyramid(motion_magnified_laplacian_pyramid);

            # Write or display the motion magnified frame.
            WriteMagnifiedFrame(motion_magnified_frame);
            # DisplayMagnifiedFrame(motion_magnified_frame);

            # Prepare for next iteration of loop
            previous_laplacian_pyramid = current_laplacian_pyramid;
            previous_riesz_x = current_riesz_x;
            previous_riesz_y = current_riesz_y;

        def ComputeRieszPyramid(grayscale_frame):
            # Compute Riesz pyramid of two dimensional frame. This is done by first
            # computing the laplacian pyramid of the frame and then computing the
            # approximate Riesz transform of each level that is not the lowpass
            # residual. The result is stored as an array of grayscale frames.
            # Corresponding locations in the result correspond to the real,
            # i and j components of Riesz pyramid coefficients.
            laplacian_pyramid = ComputeLaplacianPyramid(grayscale_frame)
            number_of_levels = numel(laplacian_pyramid)-1

            # The approximate Riesz transform of each level that is not the
            # low pass residual is computed. For more details on the approximation,
            # see supplemental material.
            kernel_x = [0.0, 0.0, 0.0,
                        0.5, 0.0, -0.5,
                        0.0, 0.0, 0.0]
            kernel_y = [0.0, 0.5, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, -0.5, 0.0]
            for k in range(1,number_of_levels):
                riesz_x[k] = Convolve(laplacian_pyramid[k], kernel_x)
                riesz_y[k] = Convolve(laplacian_pyramid[k], kernel_y)
        return laplacian_pyramid, riesz_x, riesz_y

    def ComputePhaseDifferenceAndAmplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y):
        # Computes quaternionic phase difference between current frame and previous
        # frame. This is done by dividing the coefficients of the current frame
        # and the previous frame and then taking imaginary part of the quaternionic
        # logarithm. We assume the orientation at a point is roughly constant to
        # simplify the calcuation.

        # q current = current real + i * current x + j * current y
        # q previous = previous real + i * previous x + j * previous y
        # We want to compute the phase difference, which is the phase of
        # q current/q previous
        # This is equal to (Eq. 10 of tech. report)
        # q current * conjugate(q previous) /||q previous||ˆ 2
        # Phase is invariant to scalar multiples, so we want the phase of
        # q current * conjugate(q previous)
        # which we compute now (Eq. 7 of tech. report). Under the constant orientation assumption,
        # we can assume the fourth component of the product is zero.
        q_conj_prod_real = current_real. * previous_real + current_x.*previous_x + current_y.*previous_y
        q_conj_prod_x = -current_real.*previous_x + previous_real.*current_x
        q_conj_prod_y = -current_real.*previous_y + previous_real.*current_y

        # Now we take the quaternion logarithm of this (Eq. 12 in tech. report)
        # Only the imaginary part corresponds to quaternionic phase.
        q_conj_prod_amplitude = sqrt(q_conj_prod_real.ˆ2 + q_conj_prod_x.ˆ2 + q_conj_prod_y.ˆ2)
        phase_difference = acos(q_conj_prod_real./q_conj_prod_amplitude)
        cos_orientation = q_conj_prod_x ./ sqrt(q_conj_prod_x.ˆ2+q_conj_prod_y.ˆ2)
        sin_orientation = q_conj_prod_y ./ sqrt(q_conj_prod_x.ˆ2+q_conj_prod_y.ˆ2)

        # This is the quaternionic phase (Eq. 2 in tech. report)
        phase_difference_cos = phase_difference .* cos_orientation
        phase_difference_sin = phase_difference .* sin_orientation

        # Under the assumption that changes are small between frames, we can
        # assume that the amplitude of both coefficients is the same. So,
        # to compute the amplitude of one coefficient, we just take the square root
        #
        amplitude = sqrt(q_conj_prod_amplitude);

        return phase_difference_cos, phase_difference_sin, amplitude

    def IIRTemporalFilter(B, A, phase, register0, register1):
        # Temporally filters phase with IIR filter with coefficients B, A.
        # Given current phase value and value of previously computed registers,
        # comptues current temporally filtered phase value and updates registers.
        # Assumes filter given by B, A is first order IIR filter, so that
        # B and A have 3 coefficients each. Also, assumes A(1) = 1. Computation
        # is Direct Form Type II (See pages 388-390 of Oppenheim and Schafer 3rd Ed.)
        temporally_filtered_phase = B(1) * phase + register0
        register0 = B(2) * phase + register1 - A(2) * temporally filtered phase
        register1 = B(3) * phase - A(3) * temporally filtered phase
        return temporally_filtered_phase, register0, register1

    def AmplitudeWeightedBlur(temporally filtered phase, amplitude, blur kernel):
        # Spatially blurs phase, weighted by amplitude. One half of Eq. 23 in tech. report.
        denominator = Convolve(amplitude, blur kernel)
        numerator = Convolve(temporally filtered phase.*amplitude, blur kernel)
        spatially_smooth_temporally_filtered_phase = numerator./denominator
        return spatially_smooth_temporally_filtered_phase;

    def PhaseShiftCoefficientRealPart(riesz real, riesz x, riesz y, phase cos, phase sin):
        # Phase shifts a Riesz pyramid coefficient and returns the real part of the
        # resulting coefficient. The input coefficient is a three
        # element quaternion. The phase is two element imaginary quaternion.
        # The phase is exponentiated and then the result is mutiplied by the first
        # coefficient. All operations are defined on quaternions.

        # Quaternion Exponentiation
        phase_magnitude = sqrt(phase_cos.ˆ2+phase_sin.ˆ2) # \|v\| in Eq. 11 in tech. report.
        exp_phase_real = cos(phase_magnitude)
        exp_phase_x = phase_cos./phase_magnitude.*sin(phase_magnitude)
        exp_phase_y = phase_sin./phase_magnitude.*sin(phase_magnitude)

        # Quaternion Multiplication (just real part)
        result = exp_phase_real.*riesz_real - exp_phase_x.*riesz_x - exp_phase_y.*riesz_y
        return result

    def eulerian_magnification(self, vid_data, fps, freq_min, freq_max, amplification, pyramid_levels=4, skip_levels_at_top=2):
        print("Started EVM for " + str(freq_min) + " and " + str(freq_max) + " Hz")
        vid_pyramid = self.create_laplacian_video_pyramid(vid_data, pyramid_levels=pyramid_levels)
        for i, vid in enumerate(vid_pyramid):
            if i < skip_levels_at_top or i >= len(vid_pyramid) - 1:
                # ignore the top and bottom of the pyramid. One end has too much noise and the other end is the
                # gaussian representation
                continue

            bandpassed = self.temporal_bandpass_filter(vid, fps, freq_min=freq_min, freq_max=freq_max, amplification_factor=amplification)

            # play_vid_data(bandpassed)

            vid_pyramid[i] += bandpassed
            t.sleep(1)

        vid_data = self.collapse_laplacian_video_pyramid(vid_pyramid)
        return vid_data

    def save_video(self, video, fps, save_filename='C:\\work\\output.avi'):
        """Save a video to disk"""
        # fourcc = cv2.CAP_PROP_FOURCC('M', 'J', 'P', 'G')
        print("saving file: " + save_filename)
        video = self.float_to_uint8(video)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print("video.shape ", video.shape)
        writer = cv2.VideoWriter(save_filename, fourcc, fps, (video.shape[2], video.shape[1]), 1)
        for x in range(0, video.shape[0]):
            res = cv2.convertScaleAbs(video[x])
            writer.write(res)
        writer.release()

    def float_to_uint8(self, img):
        result = np.ndarray(shape=img.shape, dtype='uint8')
        result[:] = img * 255
        return result

    def collapse_laplacian_pyramid(self, image_pyramid):
        img = image_pyramid.pop()
        while image_pyramid:
            img = cv2.pyrUp(img) + (image_pyramid.pop() - 0)

        return img

    def collapse_laplacian_video_pyramid(self, pyramid):
        i = 0
        while True:
            try:
                img_pyramid = [vid[i] for vid in pyramid]
                pyramid[0][i] = self.collapse_laplacian_pyramid(img_pyramid)
                i += 1
            except IndexError:
                break
        return pyramid[0]

    def load_video_float(self, video):
        return self.uint8_to_float(video)

    def uint8_to_float(self, img):
        result = np.ndarray(shape=img.shape, dtype='float')
        result[:] = img * (1. / 255)
        return result
