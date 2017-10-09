/* Detect a 5-CCC target, and find its pose.
*/
#include <iostream>
#include <vector>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
// This includes everything we need.
#include <opencv2/opencv.hpp>
#include "Source.h"
// Function prototypes go here.
std::vector<cv::Point2d> findTargets(cv::Mat Image);
std::vector<cv::Point2d> orderTargets(std::vector<cv::Point2d> allTargets);

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
	int frameCount = 0;
	cv::VideoCapture cap("fiveCCC.avi"); // Read a video
	if (!cap.isOpened()) { // check if we succeeded
		printf("error - can't open the camera\n");
		system("PAUSE");
		return EXIT_FAILURE;
	}	printf("Hit ESC key to quit ...\n");


	string file = "myMovie(hw3).avi";
	cv::VideoWriter output = VideoWriter(file, CV_FOURCC('M', 'P', '4', '2'), 30, Size(641, 481)); //Write Video declarations
	

	// Create 3D object model points, in the order
	// 0 1 2
	// 3 4
	std::vector<cv::Point3d> pointsModel;
	pointsModel.push_back(cv::Point3d(-3.7, -2.275, 0));
	pointsModel.push_back(cv::Point3d(0, -2.275, 0));
	pointsModel.push_back(cv::Point3d(3.7, -2.275, 0));
	pointsModel.push_back(cv::Point3d(-3.7, 2.275, 0));
	pointsModel.push_back(cv::Point3d(3.7, 2.275, 0));
	// Camera intrinsic matrix
	double K_[3][3] =
	{ { 675, 0, 320 },
	{ 0, 675, 240 },
	{ 0, 0, 1 } };
	cv::Mat K = cv::Mat(3, 3, CV_64F, K_).clone();
	cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F); // distortion coeffs

												 // Loop until no more images are available, or person hits "ESC" key.

	while (true) {
		cv::Mat oFrame;
		cv::Mat final;
		cap >> oFrame; // get image from camera or video file
		if (oFrame.empty()) break;
		// Convert to gray.
		cv::Mat imageInputGray;
		cvtColor(oFrame, imageInputGray, cv::COLOR_BGR2GRAY);
		// Find all CCC targets.
		std::vector<cv::Point2d> allTargets;
		allTargets = findTargets(imageInputGray);
		// Draw dots on all targets found.
		for (unsigned int i = 0; i < allTargets.size(); i++) {
			cv::circle(oFrame,
				allTargets[i], // center
				3, // radius
				cv::Scalar(0, 0, 255), // color
				-1); // negative thickness=filled
		}

		cv::Mat sampleImage = cv::imread("C:/sw/opencv/sources/samples/data/facebook.jpg");
		if (sampleImage.empty()) {
			std::cout << "Hey! Can't read the image!" << std::endl;
			system("PAUSE");
			return EXIT_FAILURE;
		}


		// Find the 5-CCC target pattern, and put the targets in the order
		// 1 2 3
		// 4 5
		std::vector<cv::Point2d> orderedTargets;
		orderedTargets = orderTargets(allTargets);

		if (!orderedTargets.empty()) { //Verify targets were collected else just display oFrame(original)

			vector<Point2d> sampledImage;      //Stores 4 points(x,y) of the sample image
			vector<Point2d> mainImage;    //stores 4 points(x,y) in the main image of the CCC targets

			//Orientation of sample projection is based on the order of these points
			sampledImage.push_back(cv::Point2d(double(sampleImage.cols), double(0)));
			sampledImage.push_back(cv::Point2d(double(0), double(0)));
			sampledImage.push_back(cv::Point2d(double(sampleImage.cols), double(sampleImage.rows)));
			sampledImage.push_back(cv::Point2d(double(0), double(sampleImage.rows)));


			mainImage.push_back(Point2d(orderedTargets[2].x, orderedTargets[2].y));
			mainImage.push_back(Point2d(orderedTargets[0].x, orderedTargets[0].y));
			mainImage.push_back(Point2d(orderedTargets[4].x, orderedTargets[4].y));
			mainImage.push_back(Point2d(orderedTargets[3].x, orderedTargets[3].y));


			// once we get 4 corresponding points in both images calculate homography matrix
			Mat H = findHomography(sampledImage, mainImage, 0);
			// Warp the sample image to change its perspective
			Mat warpedSample;
			warpPerspective(sampleImage, warpedSample, H, oFrame.size());

			Mat gray, gray_inv, sampleImageFinal, mainImageFinal;
			cvtColor(warpedSample, gray, CV_BGR2GRAY); //Convert to gray scale

			threshold(gray, gray, 0, 255, CV_THRESH_BINARY);
			bitwise_not(gray, gray_inv); //Invert bits

			//Create final image
			oFrame.copyTo(sampleImageFinal, gray_inv);
			warpedSample.copyTo(mainImageFinal, gray);

			final = sampleImageFinal + mainImageFinal;



			if (orderedTargets.size() == 5) {
				//Character arrays used for displaying pose numbers on image
				char frameLabel[20];

				char axLabel[20];
				char ayLabel[20];
				char azLabel[20];

				char txLabel[20];
				char tyLabel[20];
				char tzLabel[20];

				// Calculate the pose.
				cv::Mat rotVec, transVec;
				bool foundPose = cv::solvePnP(
					pointsModel, // model points
					orderedTargets, // image points
					K, // intrinsic camera parameter matrix
					dist, // distortion coefficients
					rotVec, transVec); // output rotation and translation
				if (foundPose) {
					std::vector<cv::Point3d> pointsAxes;
					std::vector<cv::Point2d> pointsImage;
					// Draw the xyz coordinate axes on the image.
					pointsAxes.push_back(cv::Point3d(0, 0, 0)); // origin
					pointsAxes.push_back(cv::Point3d(1, 0, 0)); // x axis
					pointsAxes.push_back(cv::Point3d(0, 1, 0)); // y axis
					pointsAxes.push_back(cv::Point3d(0, 0, 1)); // z axis
					cv::projectPoints(pointsAxes, rotVec, transVec, K, dist, pointsImage);
					line(final, pointsImage[0], pointsImage[1], cv::Scalar(0, 0, 255), 2);
					line(final, pointsImage[0], pointsImage[2], cv::Scalar(0, 255, 0), 2);
					line(final, pointsImage[0], pointsImage[3], cv::Scalar(255, 0, 0), 2);

					//Applying information to each label
					sprintf_s(frameLabel, "Frame Count:  %d", frameCount);

					sprintf_s(axLabel, "ax=  %f", rotVec.at<double>(0, 0));
					sprintf_s(ayLabel, "ay=  %f", rotVec.at<double>(1, 0));
					sprintf_s(azLabel, "az=  %f", rotVec.at<double>(2, 0));

					sprintf_s(txLabel, "tx=  %f", transVec.at<double>(0, 0));
					sprintf_s(tyLabel, "ty=  %f", transVec.at<double>(1, 0));
					sprintf_s(tzLabel, "tz=  %f", transVec.at<double>(2, 0));


					putText(final, frameLabel, cv::Point2d(5, 20),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(0, 0, 0), // Black
						2); // thickness
					putText(final, axLabel, cv::Point2d(5, 450),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // font color
						2); // thickness
					putText(final, ayLabel, cv::Point2d(150, 450),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // White
						2); // thickness
					putText(final, azLabel, cv::Point2d(295, 450),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // White
						2); // thickness
					putText(final, txLabel, cv::Point2d(5, 470),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // White
						2); // thickness
					putText(final, tyLabel, cv::Point2d(150, 470),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // White
						2); // thickness
					putText(final, tzLabel, cv::Point2d(295, 470),
						cv::FONT_HERSHEY_PLAIN, // font face
						1.0, // font scale
						cv::Scalar(255, 255, 255), // White
						2); // thickness
				}
			}

			namedWindow("output", WINDOW_AUTOSIZE);
			imshow("output", final);
			output.write(final);
			frameCount++;
		}
		else {
			imshow("output", oFrame);
			output.write(oFrame);
		}


		// Wait for xxx ms (0 means wait until a keypress)
		if (cv::waitKey(1) == 27) break; // hit ESC (ascii code 27) to quit }
	}
	cap.release();
	output.release();
	return EXIT_SUCCESS;
}

