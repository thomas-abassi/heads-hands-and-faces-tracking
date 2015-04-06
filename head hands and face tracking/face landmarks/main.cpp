#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp> 
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include "imgwarp_piecewiseaffine.h"
#include <thread>
#include <iostream>

using namespace cv;
using namespace dlib;
using namespace std;

// Facial head model
CascadeClassifier cascade;
// Profil head models
CascadeClassifier _cascade;
CascadeClassifier __cascade;
// Hands models
CascadeClassifier cascade2;
CascadeClassifier cascade3;
Mat menu;
Mat digits;

void detectHands(std::vector<Rect>* openhands, Mat gray);
void _detectHands(std::vector<Rect>* openhands, Mat img, CascadeClassifier _cascade);
void detectHeads2(std::vector<Rect>* faces, Mat img, bool single, bool webcam, bool fullHead, CascadeClassifier cascade, Size minSize);
void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color);
void calcSD(Mat img, Mat mask, Vec3f* mean);

int main(int argc, char** argv) {  
	try {
        // mode == 1, work on the pictures repository photos/1 ($n°photo).jpg and save results in photos2/
        // mode == 2, work on a video
        // mode == 3, work with webcam stream
		int mode = 3;
		cout << "[Tracker] Loading face and hands models" << endl;		
		cascade = CascadeClassifier("resources/xml.xml");
		_cascade = CascadeClassifier("resources/xml2.xml");
		__cascade = CascadeClassifier("resources/xml2.xml");
		cascade2 = CascadeClassifier("resources/xml3.xml");
		cascade3 = CascadeClassifier("resources/xml3.xml");
		menu = imread("resources/instructions.jpg", IMREAD_COLOR);
		digits = imread("resources/digits.png", IMREAD_GRAYSCALE);
		if (cascade.empty() || _cascade.empty() || __cascade.empty() || cascade2.empty() || cascade3.empty() || menu.empty() || digits.empty()){
			fprintf(stderr, "ERROR: Could not load models. Resources missing\n");
			return -1;
		}
        // Setting head 3D model. Another model and another algorithm could be used for better results.
		std::vector<Point3f> modelPoints;
		modelPoints.push_back(Point3f(-.2, -.25, .4));//L eye
		modelPoints.push_back(Point3f(.2, -.25, .4));//R eye
		modelPoints.push_back(Point3f(0, 0.08, 1));//Nose
		modelPoints.push_back(Point3f(0, .25, .52));//M mouth
        
        // Setting 2D/3D matching parameters
		Mat op;
		op = Mat(modelPoints);
		Scalar m = mean(Mat(modelPoints));
		double rot[9] = { 0 };
		std::vector<double> rv(3), tv(3);
		Mat rvec(rv), tvec(tv);
		Mat camMatrix;
		rvec = Mat(rv);
		double _d[9] = { 1, 0, 0,
			0, -1, 0,
			0, 0, -1 };
		Rodrigues(Mat(3, 3, CV_64FC1, _d), rvec);
		tv[0] = 0; tv[1] = 0; tv[2] = 1;
		tvec = Mat(tv);
		camMatrix = Mat(3, 3, CV_64FC1);
        // Loading facial landmarks model
        shape_predictor sp;
		String shape_predictor("resources/dat.dat");
		deserialize(shape_predictor) >> sp;
		cout << "[Tracker] Face and hands models loaded" << endl;
		cout << "[Tracker] Loading replacement faces" << endl;
        // Loading replacement faces
		std::vector<Rect> faces;
		dlib::rectangle face;
		Mat model11 = imread("models/model11.jpg", 1);
		std::vector<Point2i> pointsm11;
		Mat model12 = imread("models/model12.jpg", 1);
		std::vector<Point2i> pointsm12;
		Mat model13 = imread("models/model13.jpg", 1);
		std::vector<Point2i> pointsm13;
		Mat model21 = imread("models/model21.jpg", 1);
		std::vector<Point2i> pointsm21;
		Mat model22 = imread("models/model22.jpg", 1);
		std::vector<Point2i> pointsm22;
		Mat model23 = imread("models/model23.jpg", 1);
		std::vector<Point2i> pointsm23;
		Mat model31 = imread("models/model31.jpg", 1);
		std::vector<Point2i> pointsm31;
		Mat model32 = imread("models/model32.jpg", 1);
		std::vector<Point2i> pointsm32;
		Mat model33 = imread("models/model33.jpg", 1);
		std::vector<Point2i> pointsm33;
		Mat model41 = imread("models/model41.jpg", 1);
		std::vector<Point2i> pointsm41;
		Mat model42 = imread("models/model42.jpg", 1);
		std::vector<Point2i> pointsm42;
		Mat model43 = imread("models/model43.jpg", 1);
		std::vector<Point2i> pointsm43;
		Mat model51 = imread("models/model51.jpg", 1);
		std::vector<Point2i> pointsm51;
		Mat model52 = imread("models/model52.jpg", 1);
		std::vector<Point2i> pointsm52;
		Mat model53 = imread("models/model53.jpg", 1);
		std::vector<Point2i> pointsm53;
		Mat model61 = imread("models/model61.jpg", 1);
		std::vector<Point2i> pointsm61;
		Mat model62 = imread("models/model62.jpg", 1);
		std::vector<Point2i> pointsm62;
		Mat model63 = imread("models/model63.jpg", 1);
		std::vector<Point2i> pointsm63;
		Mat model71 = imread("models/model71.jpg", 1);
		std::vector<Point2i> pointsm71;
		Mat model72 = imread("models/model72.jpg", 1);
		std::vector<Point2i> pointsm72;
		Mat model73 = imread("models/model73.jpg", 1);
		std::vector<Point2i> pointsm73;
		Mat model81 = imread("models/model81.jpg", 1);
		std::vector<Point2i> pointsm81;
		Mat model82 = imread("models/model82.jpg", 1);
		std::vector<Point2i> pointsm82;
		Mat model83 = imread("models/model83.jpg", 1);
		std::vector<Point2i> pointsm83;
        // Detecting heads and extracting shapes
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 3; j++) {
				Mat tmp;
				if (i == 0) {
					if (j == 0) {
						tmp = model11;
					}
					else if (j == 1) {
						tmp = model12;
					}
					else if (j == 2) {
						tmp = model13;
					}
				}
				else if (i == 1) {
					if (j == 0) {
						tmp = model21;
					}
					else if (j == 1) {
						tmp = model22;
					}
					else if (j == 2) {
						tmp = model23;
					}
				}
				else if (i == 2) {
					if (j == 0) {
						tmp = model31;
					}
					else if (j == 1) {
						tmp = model32;
					}
					else if (j == 2) {
						tmp = model33;
					}
				}
				else if (i == 3) {
					if (j == 0) {
						tmp = model41;
					}
					else if (j == 1) {
						tmp = model42;
					}
					else if (j == 2) {
						tmp = model43;
					}
				}
				else if (i == 4) {
					if (j == 0) {
						tmp = model51;
					}
					else if (j == 1) {
						tmp = model52;
					}
					else if (j == 2) {
						tmp = model53;
					}
				}
				else if (i == 5) {
					if (j == 0) {
						tmp = model61;
					}
					else if (j == 1) {
						tmp = model62;
					}
					else if (j == 2) {
						tmp = model63;
					}
				}
				else if (i == 6) {
					if (j == 0) {
						tmp = model71;
					}
					else if (j == 1) {
						tmp = model72;
					}
					else if (j == 2) {
						tmp = model73;
					}
				}
				else if (i == 7) {
					if (j == 0) {
						tmp = model81;
					}
					else if (j == 1) {
						tmp = model82;
					}
					else if (j == 2) {
						tmp = model83;
					}
				}
				if (tmp.empty())  {
					break;
				}
				cv_image<bgr_pixel> _model(tmp);
				detectHeads2(&faces, tmp, true, false, true, cascade, Size(60, 60));
				face.set_top(faces[0].y);
				face.set_bottom(faces[0].y + faces[0].height);
				face.set_left(faces[0].x);
				face.set_right(faces[0].x + faces[0].width);
				full_object_detection shapem = sp(_model, face);
				std::vector<Point2i> tmp2;
				for (unsigned int k = 0; k < shapem.num_parts(); k++)  {
					tmp2.push_back(Point2i(shapem.part(k).x(), shapem.part(k).y()));
				}
				if (i == 0) {
					if (j == 0) {
						pointsm11 = tmp2;
					}
					else if (j == 1) {
						pointsm12 = tmp2;
					}
					else if (j == 2) {
						pointsm13 = tmp2;
					}
				}
				else if (i == 1) {
					if (j == 0) {
						pointsm21 = tmp2;
					}
					else if (j == 1) {
						pointsm22 = tmp2;
					}
					else if (j == 2) {
						pointsm23 = tmp2;
					}
				}
				else if (i == 2) {
					if (j == 0) {
						pointsm31 = tmp2;
					}
					else if (j == 1) {
						pointsm32 = tmp2;
					}
					else if (j == 2) {
						pointsm33 = tmp2;
					}
				}
				else if (i == 3) {
					if (j == 0) {
						pointsm41 = tmp2;
					}
					else if (j == 1) {
						pointsm42 = tmp2;
					}
					else if (j == 2) {
						pointsm43 = tmp2;
					}
				}
				else if (i == 4) {
					if (j == 0) {
						pointsm51 = tmp2;
					}
					else if (j == 1) {
						pointsm52 = tmp2;
					}
					else if (j == 2) {
						pointsm53 = tmp2;
					}
				}
				else if (i == 5) {
					if (j == 0) {
						pointsm61 = tmp2;
					}
					else if (j == 1) {
						pointsm62 = tmp2;
					}
					else if (j == 2) {
						pointsm63 = tmp2;
					}
				}
				else if (i == 6) {
					if (j == 0) {
						pointsm71 = tmp2;
					}
					else if (j == 1) {
						pointsm72 = tmp2;
					}
					else if (j == 2) {
						pointsm73 = tmp2;
					}
				}
				else if (i == 7) {
					if (j == 0) {
						pointsm81 = tmp2;
					}
					else if (j == 1) {
						pointsm82 = tmp2;
					}
					else if (j == 2) {
						pointsm83 = tmp2;
					}
				}
			}
		}
        // Extracting pure colors to use in demo
		const int ncolors = 16;
		std::vector<Scalar> colors;
		for (int n = 0; n < ncolors; ++n) {
			Mat color(Size(1, 1), CV_32FC3);
			color.at<float>(0) = (360) / ncolors * n;
			color.at<float>(1) = 1.0;
			color.at<float>(2) = 0.7;
			cvtColor(color, color, CV_HSV2BGR);
			color = color * 255;
			colors.push_back(Scalar(color.at<float>(0), color.at<float>(1), color.at<float>(2)));
		}
		cout << "[Tracker] Replacement faces loaded" << endl;

		cout << "[Tracker] Loading digits models" << endl;
        // Loading digit models
		HOGDescriptor h(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);
		std::vector<float> hdata;
		Mat data(Size(324, 5000), CV_32FC1);
		Mat responses(Size(1, 5000), CV_32FC1);
		for (int i = 0; i < 50; i++) {
			for (int j = 0; j < 100; j++) {
				h.compute(digits(Rect(j * 20, i * 20, 20, 20)), hdata);
				for (unsigned int k = 0; k < hdata.size(); k++) {
					data.at<float>(i * 100 + j, k) = hdata[k];
				}
				responses.at<float>(i * 100 + j, 0) = (float)floor(i / 5.0);
			}
		}
		KNearest knearest(data, responses);
		cout << "[Tracker] Digit models loaded" << endl;

		VideoCapture capture;
		Mat frame, gray;
		clock_t t1, t2;
		if (mode == 3) {
			cout << "[Tracker] Opening webcam" << endl;
			capture = VideoCapture(0);
			capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
			capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
			while (true) {
				capture >> frame;
				if (!frame.empty()) {
					break;
				}
			}
			cout << "[Tracker] Webcam open" << endl;
		}
		else if (mode == 2) {
			cout << "[Tracker] Loading video" << endl;
			capture = VideoCapture("v2.mov");
			while (true) {
				capture >> frame;
				if (!frame.empty()) {
					break;
				}
			}
			cout << "[Tracker] Video loaded" << endl;
		}
        
        // Variables to track heads and hands (4 max)
		std::vector<float> nbF;
		std::vector<float> nbH;
		std::vector<Rect> lastsF;
		std::vector<Rect> lastsH;
		std::vector<Point> correspH;
		std::vector<Point> correspF;
		std::vector<Point> top4H(4);
		std::vector<Point> top4F(4);
		top4H[0] = Point(-1, -1);
		top4H[1] = Point(-1, -1);
		top4H[2] = Point(-1, -1);
		top4H[3] = Point(-1, -1);
		top4F[0] = Point(-1, -1);
		top4F[1] = Point(-1, -1);
		top4F[2] = Point(-1, -1);
		top4F[3] = Point(-1, -1);
		std::vector<std::vector<Point>> traces;
		traces.push_back(std::vector<Point>(0));
		traces.push_back(std::vector<Point>(0));
		traces.push_back(std::vector<Point>(0));
		traces.push_back(std::vector<Point>(0));
		int nextIndH = 0;
        int nextIndF = 0;
        // end
        // Variable that hold what display
		int model = -1;
		bool _hands = false;
		bool freeze = false;
		bool features = false;
		bool _faces = false;
		bool fullHead = true;
		bool pose = false;
		bool lastSmile = false;
		bool lastOMouth = false;
		bool lastClosedEyes = false;
		bool displayMenu = false;
		bool trace = false;
		if (mode == 3) {
			displayMenu = true;
		}
        // end
		int sub = 1;
		long n = 1, nn = 0;
		int rate = -1;
		t1 = clock();
		float prop = .75;
		thread hand;
		int initcols, initrows;
		String name, name2;
		while (true) {
			faces.clear();
			if (mode == 1) {
				name = "photos/1 (" + to_string(n) + ").jpg";
				name2 = "photos2/1 (" + to_string(n) + ").jpg";
				frame = imread(name, 1);
				if (frame.empty()) {
					name = "photos/1 (" + to_string(n) + ").JPG";
					name2 = "photos2/1 (" + to_string(n) + ").JPG";
					frame = imread(name, 1);
					if (frame.empty()) {
						exit(0);
					}
				}
				n++;
				initcols = frame.cols / 2;
				initrows = frame.rows / 2;
			}
			else if (mode == 2) {
				capture >> frame;
				capture >> frame;
                // Resize the frame for real-time processsing
				resize(frame, frame, Size(frame.cols * 240 / frame.rows, 240));
				initcols = frame.cols;
				initrows = frame.rows;
			}
			else {
				capture >> frame;
				flip(frame, frame, 1);
				initcols = frame.cols;
				initrows = frame.rows;
			}
			int max_d = initcols;
			camMatrix = (Mat_<double>(3, 3) << max_d, 0, initcols / 2.0,
				0, max_d, initrows / 2.0,
				0, 0, 1.0);
			double _dc[] = { 0, 0, 0, 0 };
            // fullHead == true mean that face tracking algorithm use a facial and a profile model
			if (pose || _faces || model == 0) {
				fullHead = true;
			}
			else {
				fullHead = false;
			}
			std::vector<Rect> openhands;
			Mat f2;
            // tracking hands on a 3-channel image with history equalized channels
			if (_hands) {
				std::vector<Mat> bgr;
				split(frame, bgr);
				equalizeHist(bgr[0], bgr[0]);
				equalizeHist(bgr[1], bgr[1]);
				equalizeHist(bgr[2], bgr[2]);
				merge(bgr, f2);
				hand = thread(detectHands, &openhands, f2);
			}
			std::vector<dlib::rectangle> dets, dets2;
            if (_faces || features || pose || model != -1 || _hands) {
                // tracking faces on a 3-channel image with history equalized channels
				if (!_hands) {
					std::vector<Mat> bgr;
					split(frame, bgr);
					equalizeHist(bgr[0], bgr[0]);
					equalizeHist(bgr[1], bgr[1]);
					equalizeHist(bgr[2], bgr[2]);
					merge(bgr, f2);
				}
				cvtColor(f2, gray, CV_BGR2GRAY);
				cv_image<uchar> _img(gray);
				cv_image<bgr_pixel> img(f2);

                // Setting min face size and if we work with single/multiple face
				if (mode == 1) {
					detectHeads2(&faces, gray, false, false, false, cascade, Size(40, 40));
				}
				else {
					if (_faces || features || pose || model != -1) {
						detectHeads2(&faces, gray, false, true, fullHead, cascade, Size(60, 60));
					}
					else {
						detectHeads2(&faces, gray, false, true, false, cascade, Size(60, 60));

					}
				}
                // Matching new with old faces
				std::vector<Rect> lastsF2;
				std::vector<float> nbF2;
				std::vector<bool> olds, news;
				for (unsigned int i = 0; i < lastsF.size(); i++) {
					olds.push_back(false);
				}
				for (unsigned int i = 0; i < faces.size(); i++) {
					news.push_back(false);
				}
				Mat dists(Size(lastsF.size(), faces.size()), CV_32FC1, Scalar(0));
				for (unsigned int i = 0; i < lastsF.size(); i++) {
					for (unsigned int j = 0; j < faces.size(); j++) {
						dists.at<float>(j, i) = (float)sqrt(pow((lastsF[i].x + lastsF[i].width / 2) - (faces[j].x + faces[j].width / 2), 2) +
							pow((lastsF[i].y + lastsF[i].height / 2) - (faces[j].y + faces[j].height / 2), 2));
					}
				}
				double minV, maxV;
				Point minL, maxL;
				uchar ind = 0;
				correspF.clear();
				while (true) {
					if (dists.cols == 0 || dists.rows == 0) {
						break;
					}
					minMaxLoc(dists, &minV, &maxV, &minL, &maxL);
					if (nbF[minL.x] > 0) {
						if (minV < 70) {
							lastsF2.push_back(faces[minL.y]);
							if (++nbF[minL.x] >= max(4, rate / 5)) {
								nbF2.push_back(0);
								correspF.push_back(Point(minL.x, ind));
							}
							else {
								nbF2.push_back(nbF[minL.x]);
							}
							olds[minL.x] = true;
							news[minL.y] = true;
							for (int i = 0; i < dists.cols; i++) {
								dists.at<float>(minL.y, i) = maxV + 1;
							}
							for (int j = 0; j < dists.rows; j++) {
								dists.at<float>(j, minL.x) = maxV + 1;
							}
							ind++;
						}
						else {
							if (nbF[minL.x] - 1.5 > 0) {
								nbF2.push_back(nbF[minL.x] - 1.5);
								lastsF2.push_back(lastsF[minL.x]);
								olds[minL.x] = true;
								for (int j = 0; j < dists.rows; j++) {
									dists.at<float>(j, minL.x) = maxV + 1;
								}
								ind++;
							}
							else {
								olds[minL.x] = true;
								for (int j = 0; j < dists.rows; j++) {
									dists.at<float>(j, minL.x) = maxV + 1;
								}
							}
						}
					}
					else {
						lastsF2.push_back(faces[minL.y]);
						nbF2.push_back(0);
						correspF.push_back(Point(minL.x, ind));
						olds[minL.x] = true;
						news[minL.y] = true;
						for (int i = 0; i < dists.cols; i++) {
							dists.at<float>(minL.y, i) = maxV + 1;
						}
						for (int j = 0; j < dists.rows; j++) {
							dists.at<float>(j, minL.x) = maxV + 1;
						}
						ind++;
					}
					bool allOlds = true;
					bool allNews = true;
					for (unsigned int i = 0; i < olds.size(); i++) {
						if (!olds[i]) {
							allOlds = false;
							break;
						}
					}
					for (unsigned int i = 0; i < news.size(); i++) {
						if (!news[i]) {
							allNews = false;
							break;
						}
					}
					if (allNews || allOlds) {
						break;
					}
				}
				for (unsigned int i = 0; i < lastsF.size(); i++) {
					if (!olds[i]){
						if (nbF[i] <= 0) {
							nbF[i]--;
							if (nbF[i] > -max(6, rate / 4)) {
								correspF.push_back(Point(i, ind));
								nbF2.push_back(nbF[i]);
								lastsF2.push_back(lastsF[i]);
								ind++;
							}
						}
						else if (nbF[i] - 1.5 > 0) {
							nbF2.push_back(nbF[i] - 1.5);
							lastsF2.push_back(lastsF[i]);
							ind++;
						}
					}
				}
				for (unsigned int i = 0; i < faces.size(); i++) {
					if (!news[i]){
						nbF2.push_back(1);
						lastsF2.push_back(faces[i]);
					}
				}
				for (int i = 0; i < 4; i++) {
					if (top4F[i].x != -1) {
						bool found = false;
						for (unsigned int j = 0; j < correspF.size(); j++) {
							if (top4F[i].x == correspF[j].x) {
								top4F[i].x = correspF[j].y;
								found = true;
								break;
							}
						}
						if (!found) {
							top4F[i] = Point(-1, -1);
						}
					}
				}
				for (int i = 0; i < 4; i++) {
					if (top4F[i].x == -1) {
						for (unsigned int j = 0; j < correspF.size(); j++) {
							bool found = false;
							for (int k = 0; k < 4; k++) {
								if (top4F[k].x == correspF[j].y) {
									found = true;
									break;
								}
							}
							if (!found){
								top4F[i].x = correspF[j].y;
								top4F[i].y = nextIndF++;
								nextIndF %= 8;
							}
						}
					}
				}
				nbF = nbF2;
				lastsF = lastsF2;
                // end
				for (unsigned int i = 0; i < lastsF.size(); i++) {
						dlib::rectangle face;
						face.set_left(faces[i].x);
						face.set_right(faces[i].x + faces[i].width);
						face.set_top(faces[i].y);
						face.set_bottom(faces[i].y + faces[i].height);
						dets2.push_back(face);
				}
				dets = dets2;
				std::vector<full_object_detection> shapes;
                // for each face
				if (model != -1 || features || pose) {
					int nb = 0;
					for (unsigned int i = 0; i < 4; i++) {
						if (top4F[i].x != -1){
							int j = top4F[i].x;
                            // When working on pictures, alternating model used for face replacing
							if (mode == 1) {
								if (nn % 2 == 0) {
									model = 1;
								}
								else if (nn % 2 == 1) {
									model = 4;
								}
								nn++;
							}
							full_object_detection shape = sp(img, dets[j]);
                            // Draw a Delanauy triangulation
							if (model == 0) {
								Rect rect(-100, -100, frame.cols + 200, frame.rows + 200);
								Subdiv2D subdiv(rect);
								for (unsigned int k = 0; k < shape.num_parts(); k++)  {
									subdiv.insert(Point(shape.part(k).x(), shape.part(k).y()));
								}
								draw_subdiv(frame, subdiv, colors[top4F[i].y * 2 + 1]);
							}
							if (pose || features || (model > 0 && model < 9)) {
                                // Computing 6DOF head position
								std::vector<Point2f> imagePoints;
								imagePoints.push_back(Point2f((float)(shape.part(36).x()  + shape.part(37).x()  + shape.part(38).x()  +
									shape.part(39).x()  + shape.part(40).x()  + shape.part(41).x() ) / 6.0,
									(float)(shape.part(36).y()  + shape.part(37).y()  + shape.part(38).y()  + shape.part(39).y()  +
									shape.part(40).y()  + shape.part(41).y() ) / 6.0));
								imagePoints.push_back(Point2f((float)(shape.part(42).x()  + shape.part(43).x()  + shape.part(44).x()  +
									shape.part(45).x()  + +shape.part(46).x()  + shape.part(47).x() ) / 6.0,
									(float)(shape.part(42).y()  + shape.part(43).y()  + shape.part(44).y()  + shape.part(45).y()  +
									shape.part(46).y()  + shape.part(47).y() ) / 6.0));
								imagePoints.push_back(Point2f((float)shape.part(30).x() , (float)shape.part(30).y() ));
								imagePoints.push_back(Point2f((float)(shape.part(48).x()  + shape.part(54).x()  +
									shape.part(49).x()  + shape.part(59).x()  + shape.part(60).x()  + shape.part(64).x()  +
									shape.part(53).x()  + shape.part(55).x() ) / 8.0,
									(float)(shape.part(48).y()  + shape.part(54).y()  +
									shape.part(49).y()  + shape.part(59).y()  + shape.part(60).y()  + shape.part(64).y()  +
									shape.part(53).y()  + shape.part(55).y() ) / 8.0));
								Mat ip(imagePoints);
								solvePnP(op, ip, camMatrix, Mat(1, 4, CV_64FC1, _dc), rvec, tvec, true, P3P);
								rv[0] = rv[0] * 180 / 3.14 - 7;
								rv[1] = rv[1] * 180 / 3.14;
                                // Hack
								if (rv[0] < 0) {
									rv[0] *= 2;
									rv[1] *= 1.5;
								}
                                // If head tracking is on
								if (pose) {
									String rot("rotat: " + to_string((int)rv[0]) + " " + to_string((int)rv[1]) + " " + to_string((int)(rv[2] * 180 / 3.14)));
									String tra("trans: " + to_string((int)(tv[0] * 10)) + " " + to_string((int)(tv[1] * 10)) + " " + to_string((int)(tv[2] * 10)));
									putText(frame, rot, Point(dets[j].right(), dets[j].top() + nb * 15), FONT_HERSHEY_SIMPLEX, .3, colors[top4F[i].y * 2 + 1]);
									nb++;
									putText(frame, tra, Point(dets[j].right(), dets[j].top() + nb * 15), FONT_HERSHEY_SIMPLEX, .3, colors[top4F[i].y * 2 + 1]);
									nb++;
								}
							}
							float _scale = 3.5 / tv[2];
							if (model > 0 && model < 9) {
								int scale = 1;
								if (scale == 2) {
									resize(frame, frame, Size(initcols, initrows));
								}
                                // Replacing face
								Mat warpDst, warpDst2, warpDst3;
								std::vector<Point2i> dstPoints;
								long minx = frame.cols, miny = frame.rows, maxx = 0, maxy = 0;
								for (unsigned int i = 0; i < shape.num_parts(); i++)  {
									minx = min(minx, shape.part(i).x() / scale);
									miny = min(miny, shape.part(i).y() / scale);
									maxx = max(maxx, shape.part(i).x() / scale);
									maxy = max(maxy, shape.part(i).y() / scale);
								}
								if (minx >= 0 && miny >= 0 && maxx < frame.cols && maxy < frame.rows) {
									Rect ff(minx, miny, maxx - minx, maxy - miny);
									for (unsigned int i = 0; i < shape.num_parts(); i++)  {
										dstPoints.push_back(Point2i(shape.part(i).x() / scale - minx, shape.part(i).y() / scale - miny));
									}
									ImgWarp_PieceWiseAffine warpper;
									warpper.backGroundFillAlg = ImgWarp_PieceWiseAffine::BGNone;
									if (model == 1) {
										warpDst = warpper.setAllAndGenerate(model11, pointsm11, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model12.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model12, pointsm12, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model13.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model13, pointsm13, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 2) {
										warpDst = warpper.setAllAndGenerate(model21, pointsm21, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model22.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model22, pointsm22, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model23.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model23, pointsm23, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 3) {
										warpDst = warpper.setAllAndGenerate(model31, pointsm31, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model32.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model32, pointsm32, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model33.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model33, pointsm33, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 4) {
										warpDst = warpper.setAllAndGenerate(model41, pointsm41, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model42.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model42, pointsm42, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model43.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model43, pointsm43, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 5) {
										warpDst = warpper.setAllAndGenerate(model51, pointsm51, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model52.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model52, pointsm52, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model53.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model53, pointsm53, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 6) {
										warpDst = warpper.setAllAndGenerate(model61, pointsm61, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model62.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model62, pointsm62, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model63.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model63, pointsm63, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 7) {
										warpDst = warpper.setAllAndGenerate(model71, pointsm71, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model72.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model72, pointsm72, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model73.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model73, pointsm73, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									else if (model == 8) {
										warpDst = warpper.setAllAndGenerate(model81, pointsm81, dstPoints, maxx - minx, maxy - miny, 1);
										if (!model82.empty()) {
											warpDst2 = warpper.setAllAndGenerate(model82, pointsm82, dstPoints, maxx - minx, maxy - miny, 1);
											if (!model83.empty()) {
												warpDst3 = warpper.setAllAndGenerate(model83, pointsm83, dstPoints, maxx - minx, maxy - miny, 1);
											}
										}
									}
									if (!warpDst3.empty()) {
										warpDst.convertTo(warpDst, CV_32FC3);
										warpDst2.convertTo(warpDst2, CV_32FC3);
										warpDst3.convertTo(warpDst3, CV_32FC3);
										warpDst = (warpDst + warpDst2 + warpDst3) / 3;
									}
									else if (!warpDst2.empty()) {
										warpDst.convertTo(warpDst, CV_32FC3);
										warpDst2.convertTo(warpDst2, CV_32FC3);
										warpDst = (warpDst + warpDst2) / 2;
									}
									warpDst.convertTo(warpDst, CV_8UC3);
                                    // face and mouth surroundings
									std::vector<Point2f> vert;
									vert.push_back(Point(shape.part(60).x() / scale - minx, shape.part(60).y() / scale - miny));
									vert.push_back(Point(shape.part(61).x() / scale - minx, shape.part(61).y() / scale - miny));
									vert.push_back(Point(shape.part(62).x() / scale - minx, shape.part(62).y() / scale - miny));
									vert.push_back(Point(shape.part(63).x() / scale - minx, shape.part(63).y() / scale - miny));
									vert.push_back(Point(shape.part(64).x() / scale - minx, shape.part(64).y() / scale - miny));
									vert.push_back(Point(shape.part(65).x() / scale - minx, shape.part(65).y() / scale - miny));
									vert.push_back(Point(shape.part(66).x() / scale - minx, shape.part(66).y() / scale - miny));
									vert.push_back(Point(shape.part(67).x() / scale - minx, shape.part(67).y() / scale - miny));
									vert.push_back(Point(shape.part(60).x() / scale - minx, shape.part(60).y() / scale - miny));
									std::vector<Point2f> vert2;
									vert2.push_back(Point(shape.part(1).x() / scale - minx, shape.part(1).y() / scale - miny));
									vert2.push_back(Point(shape.part(2).x() / scale - minx, shape.part(2).y() / scale - miny));
									vert2.push_back(Point(shape.part(3).x() / scale - minx, shape.part(3).y() / scale - miny));
									vert2.push_back(Point(shape.part(4).x() / scale - minx, shape.part(4).y() / scale - miny));
									vert2.push_back(Point(shape.part(5).x() / scale - minx, shape.part(5).y() / scale - miny));
									vert2.push_back(Point(shape.part(6).x() / scale - minx, shape.part(6).y() / scale - miny));
									vert2.push_back(Point(shape.part(7).x() / scale - minx, shape.part(7).y() / scale - miny));
									vert2.push_back(Point(shape.part(8).x() / scale - minx, shape.part(8).y() / scale - miny));
									vert2.push_back(Point(shape.part(9).x() / scale - minx, shape.part(9).y() / scale - miny));
									vert2.push_back(Point(shape.part(10).x() / scale - minx, shape.part(10).y() / scale - miny));
									vert2.push_back(Point(shape.part(11).x() / scale - minx, shape.part(11).y() / scale - miny));
									vert2.push_back(Point(shape.part(12).x() / scale - minx, shape.part(12).y() / scale - miny));
									vert2.push_back(Point(shape.part(13).x() / scale - minx, shape.part(13).y() / scale - miny));
									vert2.push_back(Point(shape.part(14).x() / scale - minx, shape.part(14).y() / scale - miny));
									vert2.push_back(Point(shape.part(15).x() / scale - minx, shape.part(15).y() / scale - miny));
									vert2.push_back(Point(shape.part(16).x() / scale - minx, shape.part(16).y() / scale - miny));
									vert2.push_back(Point(shape.part(26).x() / scale - minx, shape.part(26).y() / scale - miny));
									vert2.push_back(Point(shape.part(25).x() / scale - minx, shape.part(25).y() / scale - miny));
									vert2.push_back(Point(shape.part(24).x() / scale - minx, shape.part(24).y() / scale - miny));
									vert2.push_back(Point(shape.part(19).x() / scale - minx, shape.part(19).y() / scale - miny));
									vert2.push_back(Point(shape.part(18).x() / scale - minx, shape.part(18).y() / scale - miny));
									vert2.push_back(Point(shape.part(17).x() / scale - minx, shape.part(17).y() / scale - miny));
                                    // Adapting the model face skin color to the target one
									Vec3f mean, mean2;
									Mat tmp, tmp2, tmp3, tmp4;
									cvtColor(warpDst, tmp4, CV_BGR2HSV);
									resize(frame(ff), tmp, Size(warpDst.cols / 3, warpDst.rows / 3));
									resize(tmp4, tmp2, Size(warpDst.cols / 3, warpDst.rows / 3));
									resize(warpDst, tmp3, Size(warpDst.cols / 3, warpDst.rows / 3));
									cvtColor(tmp, tmp, CV_BGR2HSV);
									for (int i = 0; i < tmp.cols; i++) {
										for (int j = 0; j < tmp.rows; j++) {
											tmp.at<Vec3b>(j, i)[0] = (uchar)((tmp.at<Vec3b>(j, i)[0] + 90) % 180);
											tmp2.at<Vec3b>(j, i)[0] = (uchar)((tmp2.at<Vec3b>(j, i)[0] + 90) % 180);
										}
									}
									calcSD(tmp, tmp3, &mean2);
									calcSD(tmp2, tmp3, &mean);
									Vec3b p(warpDst.at<Vec3b>(0, 0)), _p;
									mean2[0] = ((int)(mean2[0] + 90)) % 180;
									mean[0] = ((int)(mean[0] + 90)) % 180;
									for (int i = 0; i < warpDst.cols; i++) {
										for (int j = 0; j < warpDst.rows; j++) {
											_p = warpDst.at<Vec3b>(j, i);
											if (_p[0] != p[0] || _p[1] != p[1] || _p[2] != p[2]) {
												tmp4.at<Vec3b>(j, i)[0] = ((int)(mean2[0] + (tmp4.at<Vec3b>(j, i)[0] - mean[0]) + 180)) % 180;
												tmp4.at<Vec3b>(j, i)[1] = max(min(mean2[1] + (tmp4.at<Vec3b>(j, i)[1] - mean[1]), 255.0f), 0.0f);
												tmp4.at<Vec3b>(j, i)[2] = max(min(mean2[2] + (tmp4.at<Vec3b>(j, i)[2] - mean[2]), 255.0f), 0.0f);

											}
										}
									}
									cvtColor(tmp4, warpDst, CV_HSV2BGR);
                                    // Replacing
									float dist, dist2, dmax = max(6 * _scale, 3.0f),
										dmax2 = max(3 * _scale, 2.0f);
									float prop2;
									Mat crop(frame(ff));
									long cnt = 0, cnt2 = 0;
									for (int i = 0; i < maxx - minx; i++) {
										for (int j = 0; j < maxy - miny; j++) {
											_p = warpDst.at<Vec3b>(j, i);
											if (_p[0] != p[0] || _p[1] != p[1] || _p[2] != p[2]) {
                                                // Checking if point inside face and getting distance
												if (cnt % 2 == 0) {
													dist = pointPolygonTest(vert2, Point(i, j), true);
												}
												cnt++;
												if (dist > 0) {
                                                    // Checking if point outside mouth and distance
													dist2 = -pointPolygonTest(vert, Point(i, j), true);
													if (dist2 >= 0) {
														if (dist < dmax) {
															prop2 = prop * dist / dmax;
															crop.at<Vec3b>(j, i) = _p * prop2 + crop.at<Vec3b>(j, i) * (1 - prop2);
														}
														else if (dist2 < dmax2) {
															prop2 = prop * dist2 / dmax2;
															crop.at<Vec3b>(j, i) = _p * prop2 + crop.at<Vec3b>(j, i) * (1 - prop2);
														}
														else {
															crop.at<Vec3b>(j, i) = _p * prop + crop.at<Vec3b>(j, i) * (1 - prop);
														}
													}
												}
											}
										}
									}
								}
							}
                            // Computing expressions
							if (features && abs(rv[0]) < 10 && abs(rv[1]) < 10) {
								int sSmile = (shape.part(62).y() - shape.part(48).y()) + (shape.part(62).y() - shape.part(54).y());
								if (sSmile >= 2.5 * _scale) {
									if (!lastSmile) {
										lastSmile = true;
									}
									else {
										putText(frame, "Smiling", Point(dets[j].right(), dets[j].top() + nb * 15),
											FONT_HERSHEY_SIMPLEX, .3, colors[top4F[i].y * 2 + 1]);
										nb++;
									}
									lastOMouth = false;
								}
								else {
									int oMouth = sqrt(pow(shape.part(67).x() - shape.part(61).x(), 2) + pow(shape.part(67).y() - shape.part(61).y(), 2)) +
										sqrt(pow(shape.part(66).x() - shape.part(62).x(), 2) + pow(shape.part(66).y() - shape.part(62).y(), 2)) +
										sqrt(pow(shape.part(65).x() - shape.part(63).x(), 2) + pow(shape.part(65).y() - shape.part(63).y(), 2));
									if (oMouth > 27 * _scale) {
										if (lastOMouth) {
											putText(frame, "Mouth open", Point(dets[j].right(), dets[j].top() + nb * 15),
												FONT_HERSHEY_SIMPLEX, .3, colors[top4F[i].y * 2 + 1]);
											nb++;
										}
										else {
											lastOMouth = true;
										}
									}
									else {
										lastOMouth = false;
									}
									lastSmile = false;
								}
								int oEyes = sqrt(pow(shape.part(41).x() - shape.part(37).x(), 2) + pow(shape.part(41).y() - shape.part(37).y(), 2)) +
									sqrt(pow(shape.part(40).x() - shape.part(38).x(), 2) + pow(shape.part(40).y() - shape.part(38).y(), 2)) +
									sqrt(pow(shape.part(47).x() - shape.part(43).x(), 2) + pow(shape.part(47).y() - shape.part(43).y(), 2)) +
									sqrt(pow(shape.part(46).x() - shape.part(44).x(), 2) + pow(shape.part(46).y() - shape.part(44).y(), 2));
								if (oEyes <= 12 * _scale) {
									if (lastClosedEyes) {
										putText(frame, "Eyes closed", Point(dets[j].right(), dets[j].top() + nb * 15),
											FONT_HERSHEY_SIMPLEX, .3, colors[top4F[i].y * 2 + 1]);
										nb++;
									}
									else {
										lastClosedEyes = true;
									}
								}
								else {
									lastClosedEyes = false;
								}
							}
                            // Tracing an orthonormal set
							if (pose){
								rv[0] = rv[0] * 3.14 / 180;
								rv[1] = rv[1] * 3.14 / 180;
								std::vector<Point3f> objPoints;
								objPoints.push_back(Point3f(0, 0, 0));
								objPoints.push_back(Point3f(.4, 0, 0));
								objPoints.push_back(Point3f(0, -.4, 0));
								objPoints.push_back(Point3f(0, 0, .4));
								std::vector<Point2f> imgPoints;
								projectPoints(objPoints, rvec, tvec, camMatrix, Mat(1, 4, CV_64FC1, _dc), imgPoints);
								if (!freeze) {
									line(frame, imgPoints[0], imgPoints[1], Scalar(255, 0, 0), 4, CV_AA);
									line(frame, imgPoints[0], imgPoints[2], Scalar(0, 255, 0), 4, CV_AA);
									line(frame, imgPoints[0], imgPoints[3], Scalar(0, 0, 255), 4, CV_AA);
								}
							}
							shapes.push_back(shape);
						}
					}
				}
			}
			resize(frame, frame, Size(initcols * 2, initrows * 2));
			if (_hands) {
                // Matching new hands with old one
				hand.join();
				std::vector<Rect> openhands2;
				for (unsigned int i = 0; i < openhands.size(); i++) {
					bool in = false;
					for (unsigned int j = 0; j < dets.size(); j++) {
						if (openhands[i].x + openhands[i].width / 2 > dets[j].left() && openhands[i].x + openhands[i].width / 2 < dets[j].right() &&
							(openhands[i].y < dets[j].bottom() && openhands[i].y + openhands[i].height > dets[j].bottom() ||
							openhands[i].y + openhands[i].height > dets[j].top() / 2 && openhands[i].y + openhands[i].height / 2 < dets[j].bottom())) {
							in = true;
							break;
						}
					}
					if (!in) {
						openhands2.push_back(openhands[i]);
					}
				}
				openhands = openhands2;
				openhands2.clear();
				for (unsigned int i = 0; i < openhands.size(); i++) {
					bool in = false;
					for (unsigned int j = i + 1; j < openhands.size(); j++) {
						if ((openhands[i].x + openhands[i].width / 2 > openhands[j].x && openhands[i].x + openhands[i].width / 2 < openhands[j].x + openhands[j].width &&
							openhands[i].y + openhands[i].height / 2 > openhands[j].y && openhands[i].y + openhands[i].height / 2 < openhands[j].y + openhands[j].height) ||
							(openhands[j].x + openhands[j].width / 2 > openhands[i].x && openhands[j].x + openhands[j].width / 2 < openhands[i].x + openhands[i].width &&
							openhands[j].y + openhands[j].height / 2 > openhands[i].y && openhands[j].y + openhands[j].height / 2 < openhands[i].y + openhands[i].height)) {
							in = true;
							break;
						}
					}
					if (!in) {
						openhands[i].x *= 2;
						openhands[i].y *= 2;
						openhands[i].width *= 2;
						openhands[i].height *= 2;
						openhands2.push_back(openhands[i]);
					}
				}
				openhands = openhands2;
				std::vector<Rect> lastsH2;
				std::vector<float> nbH2;
				std::vector<bool> olds, news;
				for (unsigned int i = 0; i < lastsH.size(); i++) {
					olds.push_back(false);
				}
				for (unsigned int i = 0; i < openhands.size(); i++) {
					news.push_back(false);
				}
				Mat dists(Size(lastsH.size(), openhands.size()), CV_32FC1, Scalar(0));
				for (unsigned int i = 0; i < lastsH.size(); i++) {
					for (unsigned int j = 0; j < openhands.size(); j++) {
						dists.at<float>(j, i) = (float) sqrt(pow((lastsH[i].x + lastsH[i].width / 2) - (openhands[j].x + openhands[j].width / 2), 2) +
							pow((lastsH[i].y + lastsH[i].height / 2) - (openhands[j].y + openhands[j].height / 2), 2));
					}
				}
				double minV, maxV;
				Point minL, maxL;
				uchar ind = 0;
				correspH.clear();
				while (true) {
					if (dists.cols == 0 || dists.rows == 0) {
						break;
					}
					minMaxLoc(dists, &minV, &maxV, &minL, &maxL);
					if (nbH[minL.x] > 0) {
						if (minV < 70) {
							Rect newRect;
							newRect.x = lastsH[minL.x].x * .6 + openhands[minL.y].x * .4;
							newRect.y = lastsH[minL.x].y * .6 + openhands[minL.y].y * .4;
							newRect.width = lastsH[minL.x].width * .6 + openhands[minL.y].width * .4;
							newRect.height = lastsH[minL.x].height * .6 + openhands[minL.y].height * .4;
							lastsH2.push_back(newRect);
							if (++nbH[minL.x] >= max(6, rate / 4)) {
								nbH2.push_back(0);
								correspH.push_back(Point(minL.x, ind));
							}
							else {
								nbH2.push_back(nbH[minL.x]);
							}
							olds[minL.x] = true;
							news[minL.y] = true;
							for (int i = 0; i < dists.cols; i++) {
								dists.at<float>(minL.y, i) = maxV + 1;
							}
							for (int j = 0; j < dists.rows; j++) {
								dists.at<float>(j, minL.x) = maxV + 1;
							}
							ind++;
						}
						else {
							if (nbH[minL.x] - 1.5 > 0) {
								nbH2.push_back(nbH[minL.x] - 1.5);
								lastsH2.push_back(lastsH[minL.x]);
								olds[minL.x] = true;
								for (int j = 0; j < dists.rows; j++) {
									dists.at<float>(j, minL.x) = maxV + 1;
								}
								ind++;
							}
							else {
								olds[minL.x] = true;
								for (int j = 0; j < dists.rows; j++) {
									dists.at<float>(j, minL.x) = maxV + 1;
								}
							}
						}
					}
					else {
						Rect newRect;
						newRect.x = lastsH[minL.x].x * .6 + openhands[minL.y].x * .4;
						newRect.y = lastsH[minL.x].y * .6 + openhands[minL.y].y * .4;
						newRect.width = lastsH[minL.x].width * .6 + openhands[minL.y].width * .4;
						newRect.height = lastsH[minL.x].height * .6 + openhands[minL.y].height * .4;
						lastsH2.push_back(newRect);
						nbH2.push_back(0);
						correspH.push_back(Point(minL.x, ind));
						olds[minL.x] = true;
						news[minL.y] = true;
						for (int i = 0; i < dists.cols; i++) {
							dists.at<float>(minL.y, i) = maxV + 1;
						}
						for (int j = 0; j < dists.rows; j++) {
							dists.at<float>(j, minL.x) = maxV + 1;
						}
						ind++;
					}
					bool allOlds = true;
					bool allNews = true;
					for (unsigned int i = 0; i < olds.size(); i++) {
						if (!olds[i]) {
							allOlds = false;
							break;
						}
					}
					for (unsigned int i = 0; i < news.size(); i++) {
						if (!news[i]) {
							allNews = false;
							break;
						}
					}
					if (allNews || allOlds) {
						break;
					}
				}
				for (unsigned int i = 0; i < lastsH.size(); i++) {
					if (!olds[i]){
						if (nbH[i] <= 0) {
							nbH[i]--;
							if (nbH[i] > -max(6, rate / 4)) {
								correspH.push_back(Point(i, ind));
								nbH2.push_back(nbH[i]);
								lastsH2.push_back(lastsH[i]);
								ind++;
							}
						}
						else if (nbH[i] - 1.5 > 0) {
							nbH2.push_back(nbH[i] - 1.5);
							lastsH2.push_back(lastsH[i]);
							ind++;
						}
					}
				}
				for (unsigned int i = 0; i < openhands.size(); i++) {
					if (!news[i]){
						nbH2.push_back(1);
						lastsH2.push_back(openhands[i]);
					}
				}
                // Identificating numbers traced
				for (int i = 0; i < 4; i++) {
					if (top4H[i].x != -1) {
						bool found = false;
						for (unsigned int j = 0; j < correspH.size(); j++) {
							if (top4H[i].x == correspH[j].x) {
								top4H[i].x = correspH[j].y;
								found = true;
								if (trace) {
									traces[i].push_back(Point(lastsH2[top4H[i].x].x + lastsH2[top4H[i].x].width / 2,
										lastsH2[top4H[i].x].y + lastsH2[top4H[i].x].height / 2));
								}
								break;
							}
						}
						if (!found) {
							top4H[i] = Point(-1, -1);
							if (trace) {
								int minx = frame.cols, miny = frame.rows, maxx = 0, maxy = 0, maxlen;
								for (unsigned int j = 0; j < traces[i].size(); j++) {
									minx = min(minx, traces[i][j].x);
									miny = min(miny, traces[i][j].y);
									maxx = max(maxx, traces[i][j].x);
									maxy = max(maxy, traces[i][j].y);
								}			
								int sx = 0, sy = 0;
								if (maxx - minx > maxy - miny) {
									maxlen = maxx - minx;
									sy = (maxlen - (maxy - miny)) / 2;
									sy *= 14.0 / maxlen;
								}
								else {
									maxlen = maxy - miny;
									sx = (maxlen - (maxx - minx)) / 2;
									sx *= 14.0 / maxlen;
								}
								for (unsigned int j = 0; j < traces[i].size(); j++) {
									traces[i][j].x -= minx;
									traces[i][j].y -= miny;
									traces[i][j].x *= 14.0 / maxlen;
									traces[i][j].y *= 14.0 / maxlen;
									traces[i][j].x += 3 + sx;
									traces[i][j].y += 3 + sy;
								}
								Mat digit(20, 20, CV_8UC1, Scalar(0));
								for (unsigned int j = 0; j < traces[i].size() - 1; j++) {
									line(digit, traces[i][j], traces[i][j + 1], Scalar(255), 1, CV_AA);
								}
								Mat sample(Size(324, 1), CV_32FC1);
								h.compute(digit, hdata);
								for (unsigned int k = 0; k < hdata.size(); k++) {
									sample.at<float>(0, k) = hdata[k];
								}
								Mat results, neighborResponses, dists;
								cout << "Hand " << i + 1 << " traced: " << knearest.find_nearest(sample, 7, results, neighborResponses, dists) << endl;
								traces[i].resize(0);
							}
						}
					}
				}
				for (int i = 0; i < 4; i++) {
					if (top4H[i].x == -1) {
						for (unsigned int j = 0; j < correspH.size(); j++) {
							bool found = false;
							for (int k = 0; k < 4; k++) {
								if (top4H[k].x == correspH[j].y) {
									found = true;
									break;
								}
							}
							if (!found){
								top4H[i].x = correspH[j].y;
								top4H[i].y = nextIndH++;
								nextIndH %= 8;
								if (trace) {
									traces[i].push_back(Point(lastsH2[top4H[i].x].x + lastsH2[top4H[i].x].width / 2,
										lastsH2[top4H[i].x].y + lastsH2[top4H[i].x].height / 2));
								}
							}
						}
					}
				}
				nbH = nbH2;
				lastsH = lastsH2;
			}
			
            // Computing fps
			t2 = clock();
			if (rate == -1) {
				rate = ceil(1 / (((float)t2 - (float)t1) / CLOCKS_PER_SEC));
			}
			else {
				rate = rate * .75 + ceil(1 / (((float)t2 - (float)t1) / CLOCKS_PER_SEC)) * .25;
			}

			if (mode != 1) {
				putText(frame, "fps: " + to_string(rate), Point(frame.cols - 50, 15), FONT_HERSHEY_SIMPLEX, .35, CV_RGB(255, 255, 255));
			}

			if (!freeze) {
                // Sharpening image
				Mat image;
				GaussianBlur(frame, image, Size(0, 0), 3, 3);
				addWeighted(frame, 1.5, image, -0.5, 0, frame);
                // Tracing hands infos
				for (int i = 0; i < 4; i++) {
					if (top4H[i].x != -1) {
						int j = top4H[i].x;
						circle(frame, Point(lastsH[j].x + lastsH[j].width / 2, lastsH[j].y + lastsH[j].height / 2),
							40 * (lastsH[j].height / 200.0), colors[top4H[i].y * 2], -1);
						putText(frame, "x: " + to_string(lastsH[j].x + lastsH[j].width / 2 - frame.cols / 2),
							Point(lastsH[j].x + lastsH[j].width, lastsH[j].y), FONT_HERSHEY_SIMPLEX, .5, colors[top4H[i].y * 2]);
						putText(frame, "y: " + to_string(lastsH[j].y + lastsH[j].height / 2 - frame.rows / 2),
							Point(lastsH[j].x + lastsH[j].width, lastsH[j].y + 25), FONT_HERSHEY_SIMPLEX, .5, colors[top4H[i].y * 2]);
						putText(frame, "z: " + to_string(lastsH[j].height / 200.0),
							Point(lastsH[j].x + lastsH[j].width, lastsH[j].y + 50), FONT_HERSHEY_SIMPLEX, .5, colors[top4H[i].y * 2]);
						if (trace) {
							for (unsigned int j = 0; j < traces[i].size() - 1; j++) {
								line(frame, traces[i][j], traces[i][j + 1], colors[top4H[i].y * 2], 4, CV_AA);
							}
						}
					}
				}
				if (_faces) {
					for (unsigned int i = 0; i < 4; i++) {
						if (top4F[i].x != -1) {
							cv::rectangle(frame,
								Rect(dets[top4F[i].x].left() * 2, dets[top4F[i].x].top() * 2, dets[top4F[i].x].width() * 2, dets[top4F[i].x].height() * 2),
								colors[top4F[i].y * 2 + 1], 3);
						}
					}
				}
				if (displayMenu) {
					Mat tmp(frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows)));
					tmp = tmp * .33 + menu * .67;
				}
				if (mode == 1) {
					imwrite(name2, frame);
				}
				else {
					imshow("result", frame);
				}
			}
            // Demo control
			int key = waitKey(3);
			if (key == 'd') {
				freeze = !freeze;
			}
			else if (key == '0') {
				model = -1;
			}
			else if (key == '1') {
				model = 0;
			}
			else if (key == '2') {
				model = 1;
			}
			else if (key == '3') {
				model = 2;
			}
			else if (key == '4') {
				model = 3;
			}
			else if (key == '5') {
				model = 4;
			}
			else if (key == '6') {
				model = 5;
			}
			else if (key == '7') {
				model = 6;
			}
			else if (key == '8') {
				model = 7;
			}
			else if (key == '9') {
				model = 8;
			}
			else if (key == 'h') {
				_hands = !_hands;
			}
			else if (key == 'f') {
				_faces = !_faces;
			}
			else if (key == 'e') {
				features = !features;
			}
			else if (key == 'p') {
				pose = !pose;
			}
			else if (key == 'm' || key == 13) {
				displayMenu = !displayMenu;
			}
			else if (key == 'r') {
				model = -1;
				_hands = false;
				freeze = false;
				features = false;
				_faces = false;
				fullHead = true;
				pose = false;
			}
			else if (key == 't') {
				trace = !trace;
				traces[0].resize(0);
				traces[1].resize(0);
				traces[2].resize(0);
				traces[3].resize(0);
			}
			t1 = clock();
		}
    }
    catch (exception& e)
    {
        std::cout << "Sorry, exception thrown! Please restart the soft." << endl;
        std::cout << e.what() << endl;
		while (true);
    }
}


void detectHands(std::vector<Rect>* openhands, Mat gray) {
	Mat imgf;
	flip(gray, imgf, 1);
	
	std::vector<Rect> openhands2;
	thread tHands1(_detectHands, openhands, imgf, cascade2);
	thread tHands2(_detectHands, &openhands2, gray, cascade3);
	tHands1.join();
	tHands2.join();
	for (unsigned int i = 0; i < (int)(*openhands).size(); i++) {
		(*openhands)[i].x = gray.cols - (*openhands)[i].x - (*openhands)[i].width;
		openhands2.push_back((*openhands)[i]);
	}
	(*openhands) = openhands2;
}

void detectHeads2(std::vector<Rect>* faces, Mat img, bool single, bool webcam, bool fullHead, CascadeClassifier cascade, Size minSize) {
	thread rFace, lFace;
	std::vector<Rect> rFaces, lFaces;
	if (fullHead) {
		rFace = thread(detectHeads2, &rFaces, img, single, webcam, false, _cascade, minSize);
		Mat fImg;
		flip(img, fImg, 1);
		lFace = thread(detectHeads2, &lFaces, fImg, single, webcam, false, __cascade, minSize);
	}
	if (single == false && webcam == true) {
		cascade.detectMultiScale(img, *faces, 1.25, 5, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH, minSize, Size(160, 160));
	} else if (single == true && webcam == true) {
		cascade.detectMultiScale(img, *faces, 1.25, 5, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT, minSize, Size(160, 160));
	}
	else if (single == true && webcam == false) {
		cascade.detectMultiScale(img, *faces, 1.25, 5, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT, minSize);
	}
	else {
		cascade.detectMultiScale(img, *faces, 1.2, 5, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH, minSize);
	}
	for (unsigned int i = 0; i < faces->size(); i++) {
		(*faces)[i].y += (*faces)[i].height * .075;
		(*faces)[i].x += (*faces)[i].width * .075;
		(*faces)[i].height *= 1.025;
		(*faces)[i].width *= .85;
	}
	if (fullHead) {
		rFace.join();
		lFace.join();
		for (unsigned int i = 0; i < rFaces.size(); i++) {
			rFaces[i].x += rFaces[i].width * .08;
			rFaces[i].y += rFaces[i].height * .08;
			rFaces[i].width *= .84;
			rFaces[i].height *= .84;
			rFaces[i].x -= rFaces[i].width / 6;
		}
		std::vector<Rect> faces2(rFaces);
		for (unsigned int i = 0; i < lFaces.size(); i++) {
			lFaces[i].x = img.cols - lFaces[i].x - lFaces[i].width;
			lFaces[i].x += lFaces[i].width * .08;
			lFaces[i].y += lFaces[i].height * .08;
			lFaces[i].width *= .84;
			lFaces[i].height *= .84;
			lFaces[i].x += lFaces[i].width / 6;
			faces2.push_back(lFaces[i]);
		}
		for (unsigned int i = 0; i < faces->size(); i++) {
			faces2.push_back((*faces)[i]);
		}
		std::vector<Rect> faces3;
		for (unsigned int i = 0; i < faces2.size(); i++) {
			bool in = false;
			for (unsigned int j = i + 1; j < faces2.size(); j++) {
				if ((faces2[i].x + faces2[i].width / 2 > faces2[j].x && faces2[i].x + faces2[i].width / 2 < faces2[j].x + faces2[j].width &&
					faces2[i].y + faces2[i].height / 2 > faces2[j].y && faces2[i].y + faces2[i].height / 2 < faces2[j].y + faces2[j].height) ||
					(faces2[j].x + faces2[j].width / 2 > faces2[i].x && faces2[j].x + faces2[j].width / 2 < faces2[i].x + faces2[i].width &&
					faces2[j].y + faces2[j].height / 2 > faces2[i].y && faces2[j].y + faces2[j].height / 2 < faces2[i].y + faces2[i].height)) {
					in = true;
					break;
				}
			}
			if (!in) {
				faces3.push_back(faces2[i]);
			}
		}
		*faces = faces3;
	}
}

void _detectHands(std::vector<Rect>* openhands, Mat img, CascadeClassifier _cascade) {
	_cascade.detectMultiScale(img, *openhands, 1.2, 1, 0 | CV_HAAR_FEATURE_MAX | CV_HAAR_SCALE_IMAGE, Size(45, 60), Size(120, 160));
}

void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color) {
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point> pt(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		bool draw = true;
		for (int i = 0; i<3; i++){
			if (pt[i].x>img.cols || pt[i].y>img.rows || pt[i].x<0 || pt[i].y<0)
				draw = false;
		}
		if (draw){
			line(img, pt[0], pt[1], delaunay_color, 1);
			line(img, pt[1], pt[2], delaunay_color, 1);
			line(img, pt[2], pt[0], delaunay_color, 1);
		}
	}
}

void calcSD(Mat img, Mat mask, Vec3f* mean) {
	std::vector<uchar> H;
	std::vector<uchar> S;
	std::vector<uchar> V;
	int nb = 0;
	Vec3f p(mask.at<Vec3b>(0, 0)), _p;
	for (int i = 0; i < mask.cols; i++) {
		for (int j = 0; j < mask.rows; j++) {
			_p = mask.at<Vec3b>(j, i);
			if (_p[0] != p[0] || _p[1] != p[1] || _p[2] != p[2]) {
				H.push_back(img.at<Vec3b>(j, i)[0]);
				S.push_back(img.at<Vec3b>(j, i)[1]);
				V.push_back(img.at<Vec3b>(j, i)[2]);
				nb++;
			}
		}
	}
	sort(H.begin(), H.end());
	sort(S.begin(), S.end());
	sort(V.begin(), V.end());
	(*mean)[0] = H[nb / 2];
	(*mean)[1] = S[nb / 2];
	(*mean)[2] = V[nb / 2];
}