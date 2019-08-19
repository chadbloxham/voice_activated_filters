from keras.models import load_model
import cv2
import numpy as np
import speech_recognition as sr
import threading

r = sr.Recognizer()
word_input = []
word_input.append("1")


def get_audio():
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Accepting Audio Input")
            audio = r.listen(source)
            try:
                word = r.recognize_google(audio)
                word_input.insert(0, word)
                print("Input: " + word)
            except:
                pass


if __name__ == '__main__':
    t = threading.Thread(target=get_audio, name='get_audio', daemon=True)
    t.start()
    # Load the model built in the previous step
    my_model = load_model('my_model.h5')

    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Define filters
    glasses_images = ['images/sunglasses_1.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png',
           'images/sunglasses_5.jpg', 'images/sunglasses_6.png']
    glass_dict = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}

    # Hat images
    hat_images = ['images/hat_1.png', 'images/hat_2.jpg', 'images/hat_3.png']
    # Change this to change hat in frame
    hat_dict = {"1": 0, "2": 1, "3": 2, "4": 0, "5": 1, "6": 2}

    # Mustache images
    must_images = ['images/mustache_1.png', 'images/mustache_2.png', 'images/mustache_3.png']
    # Change this to change mustache in frame
    must_dict = {"1": 0, "2": 1, "3": 2, "4": 0, "5": 1, "6": 2}

    # Load the video
    camera = cv2.VideoCapture(0)

    # Keep looping
    while True:
        #filter_word = word_input[0]
        # Grab the current paintWindow
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        frame2 = np.copy(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add the 'Next Shades' button to the frame
        frame = cv2.rectangle(frame, (480, 10), (630, 43), (255, 255, 255), -1)
        filter_word = word_input[0]
        box_text = "FILTER " + filter_word
        cv2.putText(frame, box_text, (490, 37), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)
        for (x, y, w, h) in faces:

            # Grab the face
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+w]

            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255

            # Resize it to 96x96 to match the input format of the model
            original_shape = gray_face.shape # A Copy for future reference
            face_resized = cv2.resize(gray_normalized, (96, 96), interpolation=cv2.INTER_AREA)
            face_resized_copy = face_resized.copy()
            face_resized = face_resized.reshape(1, 96, 96, 1)

            # Predicting the keypoints using the model
            keypoints = my_model.predict(face_resized)

            # De-Normalize the keypoints values from (-1,1) to (0,96)
            keypoints = keypoints * 48 + 48

            # Map the Keypoints back to the original image
            face_resized_color = cv2.resize(color_face, (96, 96), interpolation=cv2.INTER_AREA)
            face_resized_color2 = np.copy(face_resized_color)

            # Pair them together
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))

            # Add shades, hat, and mustache to the frame
            glassInd = glass_dict[filter_word]
            sunglasses = cv2.imread(glasses_images[glassInd], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

            # mimic sunglasses lines for mustache
            mustInd = must_dict[filter_word]
            mustache = cv2.imread(must_images[mustInd], cv2.IMREAD_UNCHANGED)
            # (width, height) for each mustache
            must_sizes = [[int((points[11][0] - points[12][0])*1.6), int(points[13][1] - points[10][1])], [int((points[11][0] - points[12][0])*1.6), int((points[13][1] - points[10][1])*2.0)], [int((points[11][0] - points[12][0])*2.3), int((points[13][1] - points[10][1])*2.0)]]
            # (offset_y, offset_x) for each mustache
            offsets = [[int(0.2*must_sizes[mustInd][1]), int(-0.2*must_sizes[mustInd][0])], [int(0.1*must_sizes[mustInd][1]), int(-0.2*must_sizes[mustInd][0])], [int(0.05*must_sizes[mustInd][1]), int(-0.28*must_sizes[mustInd][0])]]
            must_resized = cv2.resize(mustache, (must_sizes[mustInd][0], must_sizes[mustInd][1]), interpolation=cv2.INTER_CUBIC)
            if mustInd == 0:
                transparent_region = must_resized[:, :, :3] == 0
            else:
                transparent_region = must_resized[:,:,:3] != 0
            face_resized_color[offsets[mustInd][0]+int(points[10][1]):offsets[mustInd][0]+int(points[10][1])+must_sizes[mustInd][1],offsets[mustInd][1]+int(points[12][0]):offsets[mustInd][1]+int(points[12][0])+must_sizes[mustInd][0],:][transparent_region] = must_resized[:,:,:3][transparent_region]

            # mimic sunglasses lines for hat
            hatInd = hat_dict[filter_word]
            hat = cv2.imread(hat_images[hatInd], cv2.IMREAD_UNCHANGED)
            # (width, height) for each hat
            hat_sizes = [[int(1.3*w), int(1.1*h)], [int(1.02*w), int(0.8*h)], [int(1.5*w), int(1.35*h)]]
            hat_resized = cv2.resize(hat, (hat_sizes[hatInd][0], hat_sizes[hatInd][1]), interpolation=cv2.INTER_CUBIC)
            transparent_region = hat_resized[:, :, :3] != 0

            # Resize the face_resized_color image back to its original shape
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation=cv2.INTER_CUBIC)
            # (offset_y, offset_x) for each hat
            offsets = [[int(-0.65*h), int(-0.17*w)], [int(-0.45*h), int(0.01*w)], [int(-0.7*h), int(-0.1*w)]]
            frame[offsets[hatInd][0]+y:offsets[hatInd][0]+y+hat_sizes[hatInd][1],offsets[hatInd][1]+x:offsets[hatInd][1]+x+hat_sizes[hatInd][0],:][transparent_region] = hat_resized[:, :, :3][transparent_region]

            # Show the frame
            cv2.imshow("Selfie Filters", frame)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
