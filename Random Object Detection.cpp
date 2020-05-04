/*Taimoor Daud Khan
Date: 10/12/2018
Object Tracking with SIFT
BioRobotics Lab */

#include <iostream>
#include <stdio.h>
#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\calib3d.hpp"


using namespace std;
using namespace cv;


int main(int argv, char** argc)
{
	const int MIN_MATCH_COUNT = 30;

	Mat descriptors_1;

	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	Mat trainingImg = imread("C:/Users/timur/Syncplicity/BioRobotics Laboratory/Computer Vision Algorithms/Campturing Image/image.png", 0);
	Mat frame;
	Mat Query;

	/*namedWindow("Img", cv::WINDOW_AUTOSIZE);
	imshow("Img", trainingImg);*/

	std::vector<KeyPoint> keypoints_1;
	f2d->detect(trainingImg, keypoints_1);
	f2d->compute(trainingImg, keypoints_1, descriptors_1);

	char charCheckForEscKey = 1 ;

	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		std::cout << "error: Webcam connect unsuccessful\n";
		return (0);
	}

	namedWindow("Logitech Cam", cv::WINDOW_AUTOSIZE);

	while (charCheckForEscKey != 27 && vid.isOpened())
	{
		if (!vid.read(frame))
			break;

		std::vector<KeyPoint> keypoints_2;
		Mat descriptors_2;

		cv::cvtColor(frame, Query, cv::COLOR_BGR2GRAY);
		f2d->detect(Query, keypoints_2);
		f2d->compute(Query, keypoints_2, descriptors_2);

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

		Mat img_matches;

		const float ratio_thresh = 0.75f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}

			drawMatches(trainingImg, keypoints_1, Query, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			if (good_matches.size() > MIN_MATCH_COUNT)
			{

				std::vector<Point2f> tp;
				std::vector<Point2f> qp;

				for (size_t k = 0; k < good_matches.size(); k++)
				{
					qp.push_back(keypoints_2[good_matches[k].trainIdx].pt);
					tp.push_back(keypoints_1[good_matches[k].queryIdx].pt);
				}

				Mat H = findHomography(tp, qp, RANSAC);

				cv::Size s = trainingImg.size();
				int rows = trainingImg.rows;
				int cols = trainingImg.cols;

				rows = s.height;
				cols = s.width;

				std::vector<Point2f> trainingBorder(4);
				trainingBorder[0] = cv::Point(0, 0); 
				trainingBorder[1] = cv::Point(cols-1, 0);
				trainingBorder[2] = cv::Point(cols-1, rows-1);
				trainingBorder[3] = cv::Point(0, rows-1);

				std::vector<Point2f> QueryBorder(4);

				perspectiveTransform(trainingBorder, QueryBorder, H);

				//cv::polylines(frame, QueryBorder, true, (0, 255, 0), 5);

				line(frame, QueryBorder[0] , QueryBorder[1] , Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[1] , QueryBorder[2] , Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[2] , QueryBorder[3] , Scalar(0, 255, 0), 4);
				line(frame, QueryBorder[3] , QueryBorder[0] , Scalar(0, 255, 0), 4);
				
			} 
			else
			{
				cout << "Not enough math found" << endl;
			}

		}
		
		imshow("Good Matches & Object detection", img_matches);
		imshow("Logitech Cam", frame); 

		charCheckForEscKey = cv::waitKey(1);
	}
	return (0);
}