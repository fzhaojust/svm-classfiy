#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include<iostream>

#include<fstream>
#include<sstream>
#include<string>

using namespace std;
using namespace cv;
using namespace cv::ml;
int main01()
{
	//�Ӿ���ʾ������
	int width = 512, height = 512;

	vector<vector<float>>vecdata;
	
	//��ȡѵ���������ǩ
	string th = "result.csv";

	string label = "label.csv";
	//��Ҫ������ļ� 
	ifstream iFile(th);
	ifstream isFile(label);
	string s1;

	string tmp;
	while (getline(iFile, s1))
	{
		stringstream ss(s1);
		string word;
		vector<float>v;
		while (getline(ss, word, ',')) {
			v.push_back(stof(word));
		}
		vecdata.push_back(v);
	}
	//const int size = vecdata.size();
	float trainingData[42][2];
	for (int i = 0; i < 42; ++i) {
		//cout << vecdata[i][0] << " " << vecdata[i][1] << endl;
		trainingData[i][0] = vecdata[i][0];
		trainingData[i][1] = vecdata[i][1];
		
	}
	float testData[4][2];
	for (int i = 0; i < 4; ++i) {
		testData[i][0] = vecdata[i+41][0];
		testData[i][1] = vecdata[i+41][1];
	}
	vector<int>data;
	int labels[42];
	for (int i = 0; i < 23; ++i) {
		labels[i] = 0;
	}
	for (int i = 23; i < 42; ++i) {
		labels[i] = 1;
	}


	cout << "���ݳɹ�" << endl;
	//����ѵ������
	
	//float trainingData[4][3] = { {501,10,12}, {255, 10,12}, {501, 255,255}, {10, 501,255} };

	Mat trainingDataMat(42, 2, CV_32FC1, trainingData);
	Mat labelsMat(42, 1, CV_32SC1, labels);

	//ѵ��SVM
	Ptr<SVM> svm = SVM::create();//����һ��svm����
	svm->setType(SVM::C_SVC); //����SVM��ʽ����
	svm->setKernel(SVM::RBF);//����SVM�˺�������   LINEAR
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));//����SVMѵ��ʱ������ֹ����
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);//ѵ������


	//��������
	
		
		//int response = (int)svm->predict(p);
	for (int i =0; i < 4; ++i) {
		Mat sampleMat = (Mat_<float>(1, 2) << testData[i][0], testData[i][1] );//���̸�ֵ
		float response = svm->predict(sampleMat);
		if (response == 0) {

			cout <<0 <<" ";
			
		}
		else if (response == 1) {
			cout << 1 << " ";
		
		}
	}
	

	waitKey(0);
}
