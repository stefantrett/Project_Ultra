
//date --set="27 Aug 2015 16:00:00"

//Opencv
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
//C++
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
//Mraa
#include "mraa.hpp"

using namespace cv;
using namespace std;

String face_cascade_name = "/media/card/opencv/haarcascade_frontalface_alt.xml";
String eye_cascade_name = "/media/card/opencv/haarcascade_eye.xml";

Mat faceDetect(Mat img);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

using namespace cv;
using namespace std;

enum EmotionState_t {
	HAPPY = 0,  // 0
	ANGRY,      // 1
	AMAZED,		// 2
};

static void read_csv(const string& filename, vector<Mat>& images,
		vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) {
		string error_message =
				"No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}

	string line, path, classlabel;

	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {

	// check that we are running on Galileo or Edison
	mraa_platform_t platform = mraa_get_platform_type();
	if ((platform != MRAA_INTEL_GALILEO_GEN1)
			&& (platform != MRAA_INTEL_GALILEO_GEN2)
			&& (platform != MRAA_INTEL_EDISON_FAB_C)) {
		std::cerr << "Unsupported platform, exiting" << std::endl;
		return MRAA_ERROR_INVALID_PLATFORM;
	}

	mraa::Pwm* pwm_pin_3 = new mraa::Pwm(3);
	if (pwm_pin_3 == NULL) {
		std::cerr << "Can't create mraa::Pwm object, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	mraa::Pwm* pwm_pin_5 = new mraa::Pwm(5);
	if (pwm_pin_5 == NULL) {
		std::cerr << "Can't create mraa::Pwm object, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	mraa::Pwm* pwm_pin_6 = new mraa::Pwm(6);
	if (pwm_pin_6 == NULL) {
		std::cerr << "Can't create mraa::Pwm object, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// create a gpio object from MRAA using pin 8
	mraa::Gpio* led_yellow_pin = new mraa::Gpio(8);
	if (led_yellow_pin == NULL) {
		std::cerr << "Can't create mraa::Gpio object, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// create a gpio object from MRAA using pin 9
	mraa::Gpio* led_blue_pin = new mraa::Gpio(9);
	if (led_blue_pin == NULL) {
		std::cerr << "Can't create mraa::Gpio object 9, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// create a gpio object from MRAA using pin 10
	mraa::Gpio* led_green_pin = new mraa::Gpio(10);
	if (led_green_pin == NULL) {
		std::cerr << "Can't create mraa::Gpio object 10, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// create a GPIO object from MRAA using pin 11
	mraa::Pwm* tone_pin = new mraa::Pwm(11);
	if (tone_pin == NULL) {
		std::cerr << "Can't create mraa::Pwm object 11, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// create a GPIO object from MRAA using pin 12
	mraa::Gpio* button_pin = new mraa::Gpio(12);
	if (button_pin == NULL) {
		std::cerr << "Can't create mraa::Gpio object 12, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// enable PWM on the selected pin
	if (pwm_pin_3->enable(true) != MRAA_SUCCESS) {
		std::cerr << "Cannot enable PWM on mraa::PWM object, exiting"
				<< std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	if (pwm_pin_5->enable(true) != MRAA_SUCCESS) {
		std::cerr << "Cannot enable PWM on mraa::PWM object, exiting"
				<< std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	if (pwm_pin_6->enable(true) != MRAA_SUCCESS) {
		std::cerr << "Cannot enable PWM on mraa::PWM object, exiting"
				<< std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	if (tone_pin->enable(true) != MRAA_SUCCESS) {
		std::cerr << "Cannot enable PWM on mraa::PWM object, exiting"
				<< std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// set the button as input
	if (button_pin->dir(mraa::DIR_IN) != MRAA_SUCCESS) {
		std::cerr << "Can't set digital pin as input, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	// set the pin as output
	if (led_yellow_pin->dir(mraa::DIR_OUT) != MRAA_SUCCESS) {
		std::cerr << "Can't set digital pin as output, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	if (led_blue_pin->dir(mraa::DIR_OUT) != MRAA_SUCCESS) {
		std::cerr << "Can't set digital pin as output, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	if (led_green_pin->dir(mraa::DIR_OUT) != MRAA_SUCCESS) {
		std::cerr << "Can't set digital pin as output, exiting" << std::endl;
		return MRAA_ERROR_UNSPECIFIED;
	}

	//EmotionState_t emotion;
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	argc = 2;
	argv[1] = "/media/card/emotions/my_csv.csv";
	argv[2] = "/home/root/emotions";
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}

	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading\n");
		return -1;
	};
	if (!eyes_cascade.load(eye_cascade_name)) {
		printf("--(!)Error loading\n");
		return -1;
	};

	///turn on the camera
	VideoCapture cap(-1);

	//check if the file was opened successfully
	if (!cap.isOpened()) {
		cout << "Capture could not be opened successfully" << endl;
		return -1;
	} else {
		cout << "Camera is ok.. Stay 50 cm away from your camera\n" << endl;
	}

	int w = 432;
	int h = 240;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, w);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, h);

	cout << endl << "Press the button to take a picture!" << endl;

	//test button until it's pressed
	//when it's pressed the program start to run

	int button_value = 0;
	while (!button_value) {
		button_value = button_pin->read();
		//std::cout << "value " << button_value << std::endl;
		//sleep(1);
	}

	// select a pulse width period of 1ms
	int period = 1;

	//ring 3 times
	for (int i = 0; i < 3; ++i) {
		cout << 3 - i << endl;
		tone_pin->config_percent(period, 0.5);
		usleep(50000);
		tone_pin->config_percent(period, 0);
		sleep(1);
	}

	//the frame for picture
	Mat frame;

	//ring one time, but longer
	//the tone stop after the photo is taken
	tone_pin->config_percent(period, 0.9);

	//take the picture
	cap >> frame;
	//take the picture

	//stop the tone
	tone_pin->config_percent(period, 0);

	cout << "processing the image...." << endl;

	//imwrite("image.jpg", frame);

	Mat testSample;
	testSample = faceDetect(frame);

	//imwrite("testSampe.jpg", testSample);

	// Get the path to your CSV.
	string fn_csv = string(argv[1]);

	// These vectors hold the images and corresponding labels
	vector<Mat> images;
	vector<int> labels;

	// Read in the data. This can fail if no valid
	// input filename is given.

	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg
				<< endl;
		// nothing more we can do
		exit(1);
	}

	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message =
				"This demo needs at least 2 images to  work. Please add more images to your database!";
		CV_Error(CV_StsError, error_message);
	}

	//
	//      cv::createFisherFaceRecognizer(10);
	//
	//
	//      cv::createFisherFaceRecognizer(0, 123.0);
	//

	Ptr<FaceRecognizer> model = createFisherFaceRecognizer(10);
	model->train(images, labels);

	int predictedLabel = model->predict(testSample);

	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);

	string result_message = format("Predicted class = %d", predictedLabel);
	cout << result_message << endl;

	// giving the result
	switch (predictedLabel) {
	case HAPPY:
		cout << "You are happy!" << endl;
		led_blue_pin->write(1);
		led_green_pin->write(0);
		led_yellow_pin->write(0);
		//pwm_pin_5->config_percent(1, 0.4);
		//sleep(4);
		//pwm_pin_5->config_percent(1, 0);
		system("echo 'Happy,' $(date) $(ls -1 | wc -l) >> emotions_2.txt");
		break;
	case ANGRY:
		cout << "You are angry!" << endl;
		led_blue_pin->write(0);
		led_green_pin->write(1);
		led_yellow_pin->write(0);
		//pwm_pin_3->config_percent(1, 0.4);
		//sleep(4);
		//pwm_pin_3->config_percent(1, 0);
		system("echo 'Angry,' $(date) $(ls -1 | wc -l) >> emotions_2.txt");
		break;
	case AMAZED:
		cout << "You are amazed!" << endl;
		led_blue_pin->write(0);
		led_green_pin->write(0);
		led_yellow_pin->write(1);
		//pwm_pin_6->config_percent(1, 0.4);
		//sleep(4);
		//pwm_pin_6->config_percent(1, 0);
		system("echo 'Amazed,' $(date) $(ls -1 | wc -l) >> emotions_2.txt");
		break;
	}

	cap.release();

	return 0;
}

Mat faceDetect(Mat img) {

	std::vector<Rect> faces;
	std::vector<Rect> eyes;
	bool two_eyes = false;
	bool any_eye_detected = false;

	//detecting faces
	face_cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(30, 30));

	if (faces.size() == 0) {
		cout << "Try again.. I did not dectected any faces..." << endl;
		exit(1);
	}

	Point p1 = Point(0, 0);
	for (size_t i = 0; i < faces.size(); i++) {
		// we cannot draw in the image !!! otherwise we will mess the prediction
		// rectangle( img, faces[i], Scalar( 255, 100, 0 ), 4, 8, 0 );

		Mat frame_gray;
		cvtColor(img, frame_gray, CV_BGR2GRAY);

		//imwrite("frame_gary.jpg", frame_gray);

		// croping only the face in region defined by faces[i]
		std::vector<Rect> eyes;
		Mat faceROI;
		faceROI = frame_gray(faces[i]);

		//imwrite("faceROI.jpg", faceROI);

		//In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 3,
				0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++) {
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
					faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			// we cannot draw in the image !!! otherwise we will mess the prediction
			//int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			//circle( img, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );

			if (j == 1) {
				p1 = center;
				two_eyes = true;
			} else {
				any_eye_detected = true;
			}
		}
	}

	cout << "SOME DEBUG" << endl;
	cout << "-------------------------" << endl;
	cout << "faces detected:" << faces.size() << endl;
	for (size_t j = 0; j < eyes.size(); j++) {
		cout << j << endl;
		cout << "ex: " << eyes[j].x << endl;
		cout << "ey: " << eyes[j].y << endl;
		cout << "ew: " << eyes[j].width << endl;
		cout << "eh: " << eyes[j].height << endl << endl;
	}
	cout << "x: " << faces[0].x << endl;
	cout << "y: " << faces[0].y << endl;
	cout << "w: " << faces[0].width << endl;
	cout << "h: " << faces[0].height << endl << endl;

	Mat imageInRectangle;
	imageInRectangle = img(faces[0]);
	Size recFaceSize = imageInRectangle.size();
	cout << recFaceSize << endl;

	// for debug
	//imwrite("imageInRectangle2.jpg", imageInRectangle);
	int rec_w = 0;
	int rec_h = faces[0].height * 0.64;

	// checking the (x,y) for cropped rectangle
	// based in human anatomy
	int px = 0;
	int py = 2 * 0.125 * faces[0].height;

	Mat cropImage;

	cout << "faces[0].x:" << faces[0].x << endl;
	p1.x = p1.x - faces[0].x;
	cout << "p1.x:" << p1.x << endl;

	if (any_eye_detected) {
		if (two_eyes) {
			cout << "two eyes detected" << endl;
			// we have detected two eyes
			// we have p1 and p2
			// left eye
			px = p1.x / 1.35;
		} else {
			// only one eye was found.. need to check if the
			// left or right eye
			// we have only p1
			if (p1.x > recFaceSize.width / 2) {
				// right eye
				cout << "only right eye detected" << endl;
				px = p1.x / 1.75;
			} else {
				// left eye
				cout << "only left eye detected" << endl;
				px = p1.x / 1.35;
			}
		}
	} else {
		// no eyes detected but we have a face
		px = 25;
		py = 25;
		rec_w = recFaceSize.width - 50;
		rec_h = recFaceSize.height - 30;
	}

	rec_w = (faces[0].width - px) * 0.75;
	cout << "px   :" << px << endl;
	cout << "py   :" << py << endl;
	cout << "rec_w:" << rec_w << endl;
	cout << "rec_h:" << rec_h << endl;

	cropImage = imageInRectangle(Rect(px, py, rec_w, rec_h));
	Size dstImgSize(70, 70);
	// same image size of db
	Mat finalSizeImg;
	resize(cropImage, finalSizeImg, dstImgSize);

	// for debug
	imwrite("onlyface.jpg", finalSizeImg);

	cvtColor(finalSizeImg, finalSizeImg, CV_BGR2GRAY);

	return finalSizeImg;
}

