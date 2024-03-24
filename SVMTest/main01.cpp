#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include<iostream>

#include<fstream>
#include<sstream>
#include<string>
#include <algorithm>
#include <functional>

using namespace std;
using namespace cv;
using namespace cv::ml;
//int n = 23;
//#define train_data_fresh n
//#define train_data_broken 1325
//
//#define test_data_count 4

void printVector(const vector<vector<float>> v)
{
	for (int i = 0; i < v.size(); ++i) {
		for(int j=0;j<v[i].size();++j)
		cout << v[i][j] << " ";
		cout << endl;
	}
	
}
template<class T>
void delete2dArray(T ** &x, int numberOfRows)
{
	//ɾ����ά����
	//ɾ��ÿһ�еĿռ�
	for (int i = 0; i < numberOfRows; i++)
	{
		delete[] x[i];
	}
	//ɾ����ָ��
	delete[] x;
	x = NULL;
}
//����ģ��
int main() {
	cv::Ptr<SVM> svm = SVM::load("./SVMModel.xml");
	string data01 = "dataset_AE_fresh.txt";
	string data02 = "dataset_AE_breakage.txt";
	//ifstream iFile(data01);
	ifstream iFile;
	iFile.open(data01, ios::in);
	vector<vector<float>>fresh_data;
	string s1;
	string tmp;

	//��ȡ�µ���������
	while (getline(iFile, s1))
	{
		stringstream ss(s1);
		string word;
		vector<float>v;
		while (getline(ss, word, ' ')) {
			v.push_back(stof(word));
		}
		fresh_data.push_back(v);
	}

	iFile.close();
	const int fresh_size = fresh_data.size();
	cout << fresh_data.size() << endl;
	//�����µ���ǩ
	vector<int>fresh_label;
	for (int i = 0; i < fresh_size; ++i) {
		fresh_label.push_back(0);
	}
	//for (int i = 0; i < vecdata.size(); ++i) {
	//	for (int j = 0; j < vecdata[i].size(); ++j) {
	//		cout << vecdata[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	iFile.open(data02, ios::in);
	//��ȡ������������
	vector<vector<float>>broken_data;
	while (getline(iFile, s1))
	{
		stringstream ss(s1);
		string word;
		vector<float>v;
		while (getline(ss, word, ' ')) {
			v.push_back(stof(word));
		}
		broken_data.push_back(v);
	}
	iFile.close();
	const int broken_size = broken_data.size();
	cout << broken_size << endl;
	//�������б�ǩ
	vector<int>broken_label;
	for (int i = 0; i < broken_size; ++i) {
		broken_label.push_back(1);
	}
	//�ϲ���������
	const int data_size = fresh_size + broken_size;
	vector<vector<float>> data;
	data.resize(data_size);//�ϲ�ǰ��Ҫ׼���ռ�
	merge(fresh_data.begin(), fresh_data.end(), broken_data.begin(), broken_data.end(), data.begin());
	//printVector(data);//
	//�ϲ���ǩ����
	vector<int>vec_labels;
	vec_labels.resize(data_size);
	merge(fresh_label.begin(), fresh_label.end(), broken_label.begin(), broken_label.end(), vec_labels.begin());
	int col = data[0].size();
	//����ѵ����
	float**trainingData = new float*[data_size];
	//float trainingData[train_data_fresh][2];
	for (int i = 0; i < data_size; ++i) {
		trainingData[i] = new float[col];
		for (int j = 0; j < col; ++j) {
			trainingData[i][j] = data[i][j];
		}
	}
	//����ѵ���������ǩ
	int *labels = new int[data_size];
	for (int i = 0; i < data_size; ++i) {
		labels[i] = vec_labels[i];
		
	}
	Mat trainingDataMat(data_size, col, CV_32FC1, trainingData);//���� ��� ���� ����
	int fresh_acc = 0;
	int broken_acc = 0;
	//int response = (int)svm->predict(p);
	for (int i = 0; i < data_size; ++i) {
		Mat sampleMat = (Mat_<float>(1, 8) << trainingData[i][0], trainingData[i][1], trainingData[i][2], trainingData[i][3], trainingData[i][4], trainingData[i][5], trainingData[i][6], trainingData[i][7]);//���̸�ֵ
		float response = svm->predict(sampleMat);
		if (response == 0) {
			cout << 0 << " ";
			if (i < fresh_size)fresh_acc++;
		}
		else if (response == 1) {
			cout << 1 << " ";
			if (i >= fresh_size)broken_acc++;
		}
	}

	cout << "�µ�׼ȷ��Ϊ" << float(fresh_acc) / float(fresh_size) << endl;
	cout << "����׼ȷ��Ϊ" << float(broken_acc) / float(broken_size) << endl;

	//�����ά�����ڴ�
	delete2dArray(trainingData, data_size);

	//���һά�����ڴ�
	delete[] labels;
	labels = nullptr;
	cout << "���Գɹ�" << endl;

	waitKey(0);

	return 0;
}

//ѵ��ģ��
int main22() {

	string data01 = "dataset_AE_fresh.txt";
	string data02= "dataset_AE_breakage.txt";

	//ifstream iFile(data01);
	ifstream iFile;
	iFile.open(data01, ios::in);
	vector<vector<float>>fresh_data;
	string s1;
	string tmp;
	//��ȡ�µ���������
	while (getline(iFile, s1))
	{
		stringstream ss(s1);
		string word;
		vector<float>v;
		while (getline(ss, word, ' ')) {
			v.push_back(stof(word));
		}
		fresh_data.push_back(v);
	}
	iFile.close();
	const int fresh_size = fresh_data.size();
	cout << fresh_data.size() << endl;
	//�����µ���ǩ
	vector<int>fresh_label;
	for (int i = 0; i < fresh_size; ++i) {
		fresh_label.push_back(0);
	}
	//for (int i = 0; i < vecdata.size(); ++i) {
	//	for (int j = 0; j < vecdata[i].size(); ++j) {
	//		cout << vecdata[i][j] << " ";
	//	}
	//	cout << endl;
	//}

	iFile.open(data02, ios::in);
	//��ȡ������������
	vector<vector<float>>broken_data;
	while (getline(iFile, s1))
	{
		stringstream ss(s1);
		string word;
		vector<float>v;
		while (getline(ss, word, ' ')) {
			v.push_back(stof(word));
		}
		broken_data.push_back(v);
	}
	iFile.close();
	const int broken_size = broken_data.size();
	cout << broken_size << endl;
	//�������б�ǩ
	vector<int>broken_label;
	for (int i = 0; i < broken_size; ++i) {
		broken_label.push_back(1);
	}

	//�ϲ���������
	const int data_size = fresh_size + broken_size;
	vector<vector<float>> data;
	data.resize(data_size);//�ϲ�ǰ��Ҫ׼���ռ�
	merge(fresh_data.begin(), fresh_data.end(), broken_data.begin(), broken_data.end(), data.begin());
	//printVector(data);//
	//�ϲ���ǩ����
	vector<int>vec_labels;
	vec_labels.resize(data_size);
	merge(fresh_label.begin(), fresh_label.end(), broken_label.begin(), broken_label.end(), vec_labels.begin());
	int col = data[0].size();
	//����ѵ����
		float**trainingData = new float*[data_size];
	//float trainingData[train_data_fresh][2];
		for (int i = 0; i < data_size; ++i) {
			trainingData[i] = new float[col];
			for (int j = 0; j < col; ++j) {
				trainingData[i][j] = data[i][j];
			}
		}
	//����ѵ���������ǩ
		int *labels=new int[data_size];
		for (int i = 0; i < data_size; ++i) {
			labels[i] = vec_labels[i];
			//cout << labels[i] << " ";
		}

		Mat trainingDataMat(data_size, col, CV_32FC1, trainingData);//���� ��� ���� ����
		Mat labelsMat(data_size, 1, CV_32SC1, labels);//���� ��� ���� ����
		//	//ѵ��SVM
		Ptr<SVM> svm = SVM::create();//����һ��svm����
		svm->setType(SVM::C_SVC); //����SVM��ʽ����
		svm->setKernel(SVM::RBF);//����SVM�˺�������   LINEAR SIGMOID   RBF
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));//����SVMѵ��ʱ������ֹ����
		svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);//ѵ������
		//svm->trainAuto(trainingDataMat, ROW_SAMPLE, labelsMat);
		svm->save("SVMModel.xml");
			//svm->save("SVMModel.xml");
		//	//��������
		//�������׼ȷ��
		int fresh_acc = 0;
		int broken_acc = 0;

		//int response = (int)svm->predict(p);
	for (int i = 0; i < data_size; ++i) {
		Mat sampleMat = (Mat_<float>(1, col) << trainingData[i][0], trainingData[i][1], trainingData[i][2], trainingData[i][3]);//���̸�ֵ
		//Mat sampleMat = (Mat_<float>(1, col) << trainingData[i][0], trainingData[i][1], trainingData[i][2], 
		//trainingData[i][3],trainingData[i][4], trainingData[i][5], trainingData[i][6], trainingData[i][7]);//���̸�ֵ
		float response = svm->predict(sampleMat);
		if (response == 0) {
			cout << 0 << " ";
			if (i < fresh_size)fresh_acc++;

		}
		else if (response == 1) {
			cout << 1 << " ";
			if (i >= fresh_size)broken_acc++;
		}
	}

	cout << "�µ�׼ȷ��Ϊ" << float(fresh_acc) / float(fresh_size) << endl;
	cout << "����׼ȷ��Ϊ" << float(broken_acc) /float(broken_size) << endl;

		//�����ά�����ڴ�
		delete2dArray(trainingData, data_size);

		//���һά�����ڴ�
		delete[] labels;
		labels = nullptr;
		cout << "���Գɹ�" << endl;

		waitKey(0);
		return 0;

}

//
//int main02()
//{
//	//�Ӿ���ʾ������
//	int width = 512, height = 512;
//
//	vector<vector<float>>vecdata;
//
//	//��ȡѵ���������ǩ
//	string th = "result.csv";
//
//	string label = "label.csv";
//	//��Ҫ������ļ� 
//	ifstream iFile(th);
//	ifstream isFile(label);
//	string s1;
//
//	string tmp;
//	while (getline(iFile, s1))
//	{
//		stringstream ss(s1);
//		string word;
//		vector<float>v;
//		while (getline(ss, word, ',')) {
//			v.push_back(stof(word));
//		}
//		vecdata.push_back(v);
//	}
//	//const int size = vecdata.size();
//	float trainingData[train_data_count][2];
//	for (int i = 0; i < train_data_count; ++i) {
//		//cout << vecdata[i][0] << " " << vecdata[i][1] << endl;
//		trainingData[i][0] = vecdata[i][0];
//		trainingData[i][1] = vecdata[i][1];
//
//	}
//	float testData[test_data_count][2];
//	for (int i = 0; i < test_data_count; ++i) {
//		testData[i][0] = vecdata[i + 41][0];
//		testData[i][1] = vecdata[i + 41][1];
//	}
//	vector<int>data;
//	int labels[train_data_count];
//	for (int i = 0; i < 23; ++i) {
//		labels[i] = 0;
//	}
//	for (int i = 23; i < 42; ++i) {
//		labels[i] = 1;
//	}
//
//
//	cout << "���ݳɹ�" << endl;
//	//����ѵ������
//
//	//float trainingData[4][3] = { {501,10,12}, {255, 10,12}, {501, 255,255}, {10, 501,255} };
//
//	Mat trainingDataMat(train_data_count, 2, CV_32FC1, trainingData);
//	Mat labelsMat(train_data_count, 1, CV_32SC1, labels);
//
//	//ѵ��SVM
//	Ptr<SVM> svm = SVM::create();//����һ��svm����
//	svm->setType(SVM::C_SVC); //����SVM��ʽ����
//	svm->setKernel(SVM::RBF);//����SVM�˺�������   LINEAR
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));//����SVMѵ��ʱ������ֹ����
//	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);//ѵ������
//
//
//	//��������
//
//
//		//int response = (int)svm->predict(p);
//	for (int i = 0; i < test_data_count; ++i) {
//		Mat sampleMat = (Mat_<float>(1, 2) << testData[i][0], testData[i][1]);//���̸�ֵ
//		float response = svm->predict(sampleMat);
//		if (response == 0) {
//
//			cout << 0 << " ";
//
//		}
//		else if (response == 1) {
//			cout << 1 << " ";
//
//		}
//	}
//
//
//	waitKey(0);
//}
