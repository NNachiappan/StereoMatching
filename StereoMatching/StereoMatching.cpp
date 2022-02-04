#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <thread>

#include "disparity_gpu.h"

using namespace cv;
using namespace std;

//Each thread process a section of rows
void disparityBlock(Mat &disp, short blockSize, Mat &left, Mat &right, int startI, int endI) {
	
	//loop through each pixel
	for (int i = startI; i < endI; i++) {
		for (int j = 0; j < disp.cols; j++) {

			unsigned int minSSD = numeric_limits<unsigned int>::max();
			int x = j;

			Mat lBlock = left(Rect(j, i, blockSize, blockSize)).clone();

			int thresh = floor(0.1 * disp.cols) + 1;

			//loop through each pixel on row starting from the threshold.
			for (int k = max(0, j-thresh); k <= j; k++) {

				Mat rBlock = right(Rect(k, i, blockSize, blockSize)).clone();

				Mat diff = lBlock - rBlock;

				unsigned int ssd = cv::sum(diff.mul(diff))[0];

				if (ssd < minSSD) {
					minSSD = ssd;
					x = k;
				}

			}

			disp.at<ushort>(i, j) = j - x;

		}
		cout << i << endl;
	}
}


void disparity(int numThreads, short windSize, Mat &disp, Mat &left, Mat &right) {

	//Convert to signed for Mat objects
	left.convertTo(left, CV_32S);
	right.convertTo(right, CV_32S);

	int rowSectSize = floor(disp.rows / numThreads);
	std::vector<std::thread> threads;

	for (int i = 0; i < numThreads; i++) {

		int startI = rowSectSize * i;
		int endI = rowSectSize * (i + 1);

		if (i == numThreads - 1) {
			endI = disp.rows;
		}

		thread th(disparityBlock, ref(disp), windSize, ref(left), ref(right), startI, endI);
		threads.push_back(move(th));

	}

	for (auto &th : threads) {
		th.join();
	}

}

int main()
{

	int numThreads = 4;

	std::string image_path = samples::findFile("C:/Users/Nachiappan/source/repos/StereoMatching/StereoMatching/images/kitti_1L.jpg");
	Mat left = imread(image_path, IMREAD_GRAYSCALE);
	//cv::resize(left, left, cv::Size(), 0.25, 0.25);

	image_path = samples::findFile("C:/Users/Nachiappan/source/repos/StereoMatching/StereoMatching/images/kitti_1R.jpg");
	Mat right = imread(image_path, IMREAD_GRAYSCALE);
	//cv::resize(right, right, cv::Size(), 0.25, 0.25);

	int windSize = 0;
	cout << "Enter window size:" << endl;
	cin >> windSize;


	Mat disp = Mat::zeros(left.rows, left.cols, CV_16U);
	short border = floor(windSize / 2);

	left.convertTo(left, CV_8U);
	right.convertTo(right, CV_8U);

	copyMakeBorder(left, left, border, border, border, border, BORDER_REPLICATE);
	copyMakeBorder(right, right, border, border, border, border, BORDER_REPLICATE);


	//disparity(numThreads, windSize, disp, left, right);
	disparityGPU(disp, left, right, disp.rows, disp.cols, windSize);


	disp.convertTo(disp, CV_32F);
	normalize(disp, disp, 0, 1, NORM_MINMAX, -1);

	imshow("Disparity", disp);
	waitKey(0);


	return 0;
}
