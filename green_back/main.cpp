#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int createMaskKmeans(cv::Mat src, cv::Mat &mask)
{
	if ((mask.type() != CV_8UC1) || (src.size() != mask.size())) {
		return 0;
	}
	int width = src.cols;
	int height = src.rows;
	int pixNum = width*height;
	int clusterCount = 2;
	Mat labels;
	Mat centers;

	Mat sampleData = src.reshape(3, pixNum);
	Mat km_data;
	sampleData.convertTo(km_data, CV_32F);

	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(km_data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);
	uchar fg[2] = { 0,255 };
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			mask.at<uchar>(row, col) = fg[labels.at<int>(row*width + col)];
		}
	}

	return 0;
}

int main()
{
	VideoCapture cap("cat.mov");
	Mat backgroud1 = imread("test1.jpg");
	resize(backgroud1, backgroud1, Size(640, 360));
	while (1) {
		Mat frame;
		cap >> frame;
		Mat frame1;
		Mat backgroud = imread("test1.jpg");
		resize(backgroud, backgroud, Size(640, 360));
		resize(frame, frame, Size(640, 360));
		frame.copyTo(frame1);
		Mat mask = Mat::zeros(frame.size(), CV_8UC1);
		createMaskKmeans(frame, mask);
		/*以第一个点为参考点，作为背景的参考*/
		if (mask.at<uchar>(0, 0) == 0) {
			for (int i = 0; i < mask.rows; i++) {
				for (int j = 0; j < mask.cols; j++) {
					mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);
				}

			}
		}

		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				if (mask.at<uchar>(i, j) != 0) {
					frame.at<Vec3b>(i, j)[0] = 0;
					frame.at<Vec3b>(i, j)[1] = 0;
					frame.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					backgroud.at<Vec3b>(i, j)[0] = 0;
					backgroud.at<Vec3b>(i, j)[1] = 0;
					backgroud.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				backgroud.at<Vec3b>(i, j)[0] = backgroud.at<Vec3b>(i, j)[0] + frame.at<Vec3b>(i, j)[0];
				backgroud.at<Vec3b>(i, j)[1] = backgroud.at<Vec3b>(i, j)[1] + frame.at<Vec3b>(i, j)[1];
				backgroud.at<Vec3b>(i, j)[2] = backgroud.at<Vec3b>(i, j)[2] + frame.at<Vec3b>(i, j)[2];
			}
		}
		imshow("cat", frame1);
		imshow("backgroud", backgroud1);
		imshow("result", backgroud);
		cv::waitKey(30);
		}
	}