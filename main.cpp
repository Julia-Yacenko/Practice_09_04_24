#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier smiles_cascade;

int main()
{
	if (!face_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_frontalface_alt.xml"))) {
		printf("Error loading face cascade model \n"); 
		return -1;
	}

	if (!eyes_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_eye_tree_eyeglasses.xml"))) {
		printf("Error loading face cascade model \n");
		return -1;
	}

	if (!smiles_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_smile.xml"))) {
		printf("Error loading face cascade model \n");
		return -1;
	}

	VideoCapture cap("video_face.mp4");
	if (!cap.isOpened()) {
		cout << "Error opening video" << endl;
		return -1;
	}

	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	VideoWriter video("./outface.mp4", cv::VideoWriter::fourcc('A', 'V', 'C', '1'), cap.get(cv::CAP_PROP_FPS), cv::Size(frame_width * 0.6, frame_height * 0.6));

	namedWindow("Output Video");

	Mat Img;
	while (true) {
		cap >> Img;
		if (Img.empty())
		{
			break;
		}

		/*Mat Img = cv::imread("D:/Camera/face.png");
		if (!Img.data)
		{
			printf("Error loading image \n");
			return -1;
		}*/

		Mat origImg = Img.clone();
		cv::resize(origImg, Img, cv::Size(), 0.6, 0.6);

		Mat image1;
		GaussianBlur(Img, image1, Size(3, 3), 0);

		Mat grayImg;
		cvtColor(image1, grayImg, COLOR_BGR2GRAY);

		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(grayImg, faces, 1.1, 5);

		for (const auto& face : faces) {
			rectangle(Img, face, Scalar(255, 0, 0), 2);
		}

		std::vector<cv::Rect> eyes;
		eyes_cascade.detectMultiScale(grayImg, eyes, 1.1, 5);

		for (const auto& eye : eyes) {
			rectangle(Img, eye, Scalar(0, 255, 0), 2);
		}

		std::vector<cv::Rect> smiles;
		smiles_cascade.detectMultiScale(grayImg, smiles, 1.165, 35, 0, cv::Size(25, 25));

		for (const auto& smile : smiles) {
			rectangle(Img, smile, Scalar(0, 0, 255), 2);
		}
	
		video.write(Img);
		imshow("Output Video", Img);
		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	cap.release();
	video.release();

	destroyAllWindows();

	return 0;

}
