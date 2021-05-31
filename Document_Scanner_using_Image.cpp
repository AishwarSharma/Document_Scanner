#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include<iostream>

using namespace std;
using namespace cv;

////////////////////////////// DOCUMENT SCANNER ///////////////

/// <summary>
/// 1) First we will Pre-Process the Image.(grayscale->Blur->Edge detection).
///    * With edge detection we can find where our paper is and based on 
///		 the coordinates we will extract 4 points of vertices. At last we will warp image.
/// 2) Getting Contours - Biggest Rectangle in image(A4 size paper).
/// 3) Warp
/// </summary>

Mat imgOriginal, imgGray, imgCanny, imgThreshold, imgBlur, imgDil ,imgWarp,imgCrop;
vector<Point> initialPoints ,docPoints;
float w = 420, h = 596;

Mat preProcessing(Mat img)
{
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);

	return imgDil;
}

vector<Point> getContours(Mat imgDil)
{
	vector<vector<Point>> contours; // 2D vector of Points, where Points is a type, probably a class
	vector<Vec4i> hierarchy; //A Vec4i is a type holding 4 integers.

	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea=0;

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);   // Calculates area of contours
		//cout << area << endl;

		if (area > 1000)
		{
			// Finding the bounding boxes around objects.

			float peri = arcLength(contours[i], true);  // true means object is closed
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);   // Finding the number of curvers(corners) the polygon has.

			if (area > maxArea && conPoly[i].size() == 4)
			{
				//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);  
				biggest = { conPoly[i][0],conPoly[i][1],conPoly[i][2],conPoly[i][3] };   
				maxArea = area;
			}

		}
	}
	return biggest;
}

void drawPoints(vector<Point> points, Scalar color)
{
	for (int i = 0; i < points.size(); i++)
	{
		circle(imgOriginal, points[i], 10, color, FILLED);
		putText(imgOriginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, color, 2);
	}
}

vector<Point> reorder(vector<Point> points)
{
	vector<Point> newPoints;
	vector<int> sumPoints , subPoints;
	
	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);

	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);  // 0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);  // 1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);  // 2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);  // 3 

	return newPoints;

}

Mat getWarp(Mat imgOriginal,vector<Point> docPoints,float w, float h)
{
	Point2f src[4] = { docPoints[0],docPoints[1],docPoints[2],docPoints[3] };
	Point2f dst[4] = { { 0.0f,0.0f }, { w,0.0f }, { 0.0f,h }, { w,h } };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(imgOriginal, imgWarp, matrix, Point(w, h));

	return imgWarp;
}	

void main()
{
	string path = "Resources/paper.jpg";
	imgOriginal = imread(path);
	resize(imgOriginal, imgOriginal, Size(), 0.5, 0.5);
		
	/* Pre-Processing: */
	imgThreshold = preProcessing(imgOriginal);

	/* Get Contours - Biggest Rectangle. */
	initialPoints = getContours(imgThreshold);
	//drawPoints(initialPoints, Scalar(0,0,255));
	docPoints = reorder(initialPoints);
	//drawPoints(docPoints, Scalar(0, 255, 0));

	/* Warp */
	imgWarp = getWarp(imgOriginal,docPoints,w,h);

	/*Crop*/
	int cropVal = 5;
	Rect roi(cropVal, cropVal, w - (2* cropVal), h - (2* cropVal));
	imgCrop = imgWarp(roi);

	imshow("Image", imgOriginal);
	imshow("Image Dilation", imgThreshold);
	imshow("Image Warp", imgWarp);
	imshow("Image Crop", imgCrop);
	waitKey(0);

}