import numpy as np
import time
import cv2
import pylab
import os
import sys


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
        vals = self.get_subface_means(self.getForehead())

        self.data_buffer.append(vals)
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

        self.drawMenu(str(cam))
        self.drawFaceRect()

        if self.find_faces:
            # face detection not yet done
            self.findFaces()
            return

        dataGathered = self.gatherColorIntensityData()
        processed = np.array(self.data_buffer)
        self.samples = processed
        buffer_len = len(self.data_buffer)

        if len(self.times) > 1:
            self.fps = float(buffer_len) / (self.times[-1] - self.times[0])
            self.gap = (self.buffer_size - buffer_len) / self.fps

        if not dataGathered:
            return

        # evenly space numbers from beginning to current time - on L spaces
        even_times = np.linspace(self.times[0], self.times[-1], buffer_len)

        # interpolate even_times to processed based on self.times->processed map
        interpolated = np.interp(even_times, self.times, processed)

        # apply Hamming window function (for L points in the result)
        # to select only the area of interest in the signal and smoothen it
        interpolated = self.hamming_window * interpolated

        # remove mean value from interpolated (simple denoise)
        interpolated = interpolated - np.mean(interpolated)

        # Fourier transformation to amplify the signal
        rawfft = np.fft.rfft(interpolated)

        # convert transformed signal to radians
        phase = np.angle(rawfft)
        # remove negative values from the transformed signal
        self.fft = np.abs(rawfft)

        self.freqs = float(self.fps) / buffer_len * np.arange(buffer_len / 2 + 1)
        self.freqs = 60. * self.freqs # freqs per minute

        idx = np.where((self.freqs > 40) & (self.freqs < 150))

        # select only that area of the signal (in radians) where freqs are in target bpm limits
        self.fft = self.fft[idx]
        phase = phase[idx]
        self.freqs = self.freqs[idx]

        max_val_indices = np.argmax(self.fft)

        # get sin of signal in radians [-1;1], and project values to [0;1]
        t = (np.sin(phase[max_val_indices]) + 1.) / 2.
        # split value to two compounds: alpha [0.1;1] and beta [0;0.9]
        # alpha+beta=1:
        #    the smaller is alpha, the bigger is beta and vice versa
        #    the bigger is t, the bigger is alpha
        #    beta will show maximums from sin, and will be added to the green channel
        t = 0.9 * t + 0.1
        alpha = t
        beta = 1 - t

        self.bpm = self.freqs[max_val_indices]

        x, y, w, h = self.getForehead()
        b = alpha * self.frame_in[y:y + h, x:x + w, 0]
        g = alpha * self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
        r = alpha * self.frame_in[y:y + h, x:x + w, 2]
        self.frame_out[y:y + h, x:x + w] = cv2.merge([b,g,r])

        # copy image to the plot window
        face_x, face_y, face_w, face_h = self.face_rect
        self.slices = [
                np.copy(
                        self.frame_out[
                                face_y:face_y + face_h,
                                face_x:face_x + face_w,
                                1
                            ]
                    )
            ]
