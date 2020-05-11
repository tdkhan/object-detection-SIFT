/*
MIT License

Copyright (c) 2020 Taimoor Daud Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
	
/*
Taimoor Daud Khan
Date: 10/12/2018
Object Tracking with SIFT
*/

#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\calib3d.hpp"

using namespace std;
using namespace cv;

int main(int argv, char** argc)
{
	const int MIN_MATCH_COUNT = 30; // minimum number of SIFT features for successful object detection

	// vector of keypoints to hold the feature coordinates & a mat object to hold the feature descriptors (training image)
	Mat descriptors_1;
	vector<KeyPoint> keypoints_1;

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(); // pointer to 2D SIFT features extractor object

	Mat trainingImg = imread("C:/Users/timur/Syncplicity/BioRobotics Laboratory/Computer Vision Algorithms/Object Tracking SIFT/image.png", 0); // load the cropped grey-scale image of the objet of interest

	// Mat objects for camera frames
	Mat frame;
	Mat Query;

	// uncomment the following lines to display the training image
	/*namedWindow("Training Img", WINDOW_AUTOSIZE);
	imshow("Training Img", trainingImg);*/

	// detect and compute SIFT features from the training image, returns the feature keypoints and descriptors
	f2d->detect(trainingImg, keypoints_1);
	f2d->compute(trainingImg, keypoints_1, descriptors_1);

	// check for features in the training image
	if (descriptors_1.empty())
	{
		cout << "No features detected in the training image!" << endl;
	}

	// define a key for program termination and initialize it to any character other than the one used for termination
	char charCheckForEscKey = 1 ;

	VideoCapture vid(0); // an object for video capturing

	if (!vid.isOpened())
	{
		cout << "error: Webcam connect unsuccessful\n";
		return (0);
	}

	namedWindow("Logitech Cam", WINDOW_AUTOSIZE); // define a window for hosting video

	// loop runs till we exit by pressing Esc or untill the camera disconnects
	while (charCheckForEscKey != 27 && vid.isOpened())
	{
		if (!vid.read(frame))
			break;

		// vector of keypoints to hold the feature coordinates & a mat object to hold the feature descriptors (current video frame)
		vector<KeyPoint> keypoints_2;
		Mat descriptors_2;

		cvtColor(frame, Query, COLOR_BGR2GRAY); // grey-scale image of current video frame

		// detect and compute SIFT features from the Query frame, returns the feature keypoints and descriptors
		f2d->detect(Query, keypoints_2);
		f2d->compute(Query, keypoints_2, descriptors_2);

		// pointer to a SIFT descriptor matcher object
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

		// check if features are detected in both the training and query images
		if (!descriptors_2.empty() && !descriptors_1.empty())
		{

			// find 2 best matches for each descriptor in the Query frame
			vector<vector<DMatch>> knn_matches;
			matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

			// image frame to show SIFT feature detection
			Mat img_matches;

			const float ratio_thresh = 0.75f;
			vector<DMatch> good_matches;


			// loop over the matches found
			for (size_t i = 0; i < knn_matches.size(); i++)
			{
				// filter for good matches using ratio test
				if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
				{
					good_matches.push_back(knn_matches[i][0]);
				}
			}

			// draw detected features in frame
			drawMatches(trainingImg, keypoints_1, Query, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


			if (good_matches.size() > MIN_MATCH_COUNT)
			{

				// vector for training and query keypoints for good matches
				vector<Point2f> tp;
				vector<Point2f> qp;

				for (size_t k = 0; k < good_matches.size(); k++)
				{
					qp.push_back(keypoints_2[good_matches[k].trainIdx].pt);
					tp.push_back(keypoints_1[good_matches[k].queryIdx].pt);
				}

				// homograph transform
				Mat H = findHomography(tp, qp, RANSAC);

				// training image size
				Size s = trainingImg.size();
				int rows = s.height;
				int cols = s.width;

				// training object vertices starting from top left and then moving clockwise
				vector<Point2f> trainingBorder(4);
				trainingBorder[0] = Point(0, 0);
				trainingBorder[1] = Point(cols - 1, 0);
				trainingBorder[2] = Point(cols - 1, rows - 1);
				trainingBorder[3] = Point(0, rows - 1);

				// query object vertices
				vector<Point2f> QueryBorder(4);
				perspectiveTransform(trainingBorder, QueryBorder, H);

				// draw a green colored border around the object
				line(frame, QueryBorder[0], QueryBorder[1], Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[1], QueryBorder[2], Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[2], QueryBorder[3], Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[3], QueryBorder[0], Scalar(0, 255, 0), 4);

				// image with detected features
				imshow("Good Matches & Object detection", img_matches);

			}
			else
			{
				cout << "Not enough matches found" << endl;
			}
		}

		// display the video frame
		imshow("Logitech Cam", frame);

		// wait 1ms to check for Esc key
		charCheckForEscKey = waitKey(1);
	}

	return (0);
}
