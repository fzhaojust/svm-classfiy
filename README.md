# 基于opencv的SVM类进行的，台湾大学林智仁开发的LIBSVM开发包的。实现了针对AE信号提取8类有效特征的二分类问题， #
### 特征txt数据集存于文件内，请自行查看
### 觉得代码简单易用的，请给个star哟，爱你  ###
## 改进点 ##
将数据修改为动态数组，实现了对特征维数的自动适配，无需手动修改，便于维护。

# 设置SVM参数 #


struct CvSVMParams

SVM 训练参数结构。

该结构必须被初始化后，传给CvSVM。

CvSVMParams::CvSVMParams
构造函数

C++: CvSVMParams::CvSVMParams()
C++: CvSVMParams::CvSVMParams(int svm_type, int kernel_type, double degree, double gamma, double coef0, double Cvalue, double nu, double p, CvMat* class_weights, CvTermCriteria term_crit)
参数	
svm_type –
指定SVM的类型，下面是可能的取值：

CvSVM::C_SVC C类支持向量分类机。 n类分组  (n \geq 2)，允许用异常值惩罚因子C进行不完全分类。
CvSVM::NU_SVC \nu类支持向量分类机。n类似然不完全分类的分类器。参数为 \nu 取代C（其值在区间【0，1】中，nu越大，决策边界越平滑）。
CvSVM::ONE_CLASS 单分类器，所有的训练数据提取自同一个类里，然后SVM建立了一个分界线以分割该类在特征空间中所占区域和其它类在特征空间中所占区域。
CvSVM::EPS_SVR \epsilon类支持向量回归机。训练集中的特征向量和拟合出来的超平面的距离需要小于p。异常值惩罚因子C被采用。
CvSVM::NU_SVR \nu类支持向量回归机。 \nu 代替了 p。
可从 [LibSVM] 获取更多细节。

kernel_type –
SVM的内核类型，下面是可能的取值：

CvSVM::LINEAR 线性内核。没有任何向映射至高维空间，线性区分（或回归）在原始特征空间中被完成，这是最快的选择。K(x_i, x_j) = x_i^T x_j.
CvSVM::POLY 多项式内核： K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0.
CvSVM::RBF 基于径向的函数，对于大多数情况都是一个较好的选择： K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0.
CvSVM::SIGMOID Sigmoid函数内核：K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).
degree – 内核函数（POLY）的参数degree。
gamma – 内核函数（POLY/ RBF/ SIGMOID）的参数\gamma。
coef0 – 内核函数（POLY/ SIGMOID）的参数coef0。
Cvalue – SVM类型（C_SVC/ EPS_SVR/ NU_SVR）的参数C。
nu – SVM类型（NU_SVC/ ONE_CLASS/ NU_SVR）的参数 \nu。
p – SVM类型（EPS_SVR）的参数 \epsilon。
class_weights – C_SVC中的可选权重，赋给指定的类，乘以C以后变成 class\_weights_i * C。所以这些权重影响不同类别的错误分类惩罚项。权重越大，某一类别的误分类数据的惩罚项就越大。
term_crit – SVM的迭代训练过程的中止条件，解决部分受约束二次最优问题。您可以指定的公差和/或最大迭代次数。
默认的构造函数初始化有以下值：

复制代码
CvSVMParams::CvSVMParams() :
    svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0),
    gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
}
复制代码
 

OpenCV的SVM 

class CvSVM

向量支持机

CvSVM::CvSVM
训练构造函数。

C++: CvSVM::CvSVM()
C++: CvSVM::CvSVM(const Mat& trainData, const Mat& responses, const Mat& varIdx=Mat(), const Mat& sampleIdx=Mat(), CvSVMParamsparams=CvSVMParams() )
C++: CvSVM::CvSVM(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, const CvMat* sampleIdx=0, CvSVMParamsparams=CvSVMParams() )
参数	
trainData — 训练数据，必须是CV_32FC1 （32位浮点类型，单通道）。数据必须是CV_ROW_SAMPLE的，即特征向量以行来存储。
responses — 响应数据，通常是1D向量存储在CV_32SC1 （仅仅用在分类问题上）或者CV_32FC1格式。
varIdx — 指定感兴趣的特征。可以是整数(32sC1)向量，例如以0为开始的索引，或者8位(8uC1)的使用的特征或者样本的掩码。用户也可以传入NULL指针，用来表示训练中使用所有变量／样本。
sampleIdx — 指定感兴趣的样本。描述同上。
params — SVM参数。
CvSVM::train
训练一个SVM。

C++: bool CvSVM::train(const Mat& trainData, const Mat& responses, const Mat& varIdx=Mat(), const Mat& sampleIdx=Mat(), CvSVMParamsparams=CvSVMParams() )
C++: bool CvSVM::train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, const CvMat* sampleIdx=0, CvSVMParamsparams=CvSVMParams() )
参数参考构造函数。

CvSVM::train_auto
根据可选参数训练一个SVM。

C++: bool CvSVM::train_auto(const Mat& trainData, const Mat& responses, const Mat& varIdx, const Mat& sampleIdx, CvSVMParamsparams, int k_fold=10, CvParamGrid Cgrid=CvSVM::get_default_grid(CvSVM::C), CvParamGridgammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA), CvParamGrid pGrid=CvSVM::get_default_grid(CvSVM::P), CvParamGridnuGrid=CvSVM::get_default_grid(CvSVM::NU), CvParamGrid coeffGrid=CvSVM::get_default_grid(CvSVM::COEF), CvParamGriddegreeGrid=CvSVM::get_default_grid(CvSVM::DEGREE), bool balanced=false)
C++: bool CvSVM::train_auto(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx, const CvMat* sampleIdx, CvSVMParams params, int kfold=10, CvParamGrid Cgrid=get_default_grid(CvSVM::C), CvParamGrid gammaGrid=get_default_grid(CvSVM::GAMMA), CvParamGrid pGrid=get_default_grid(CvSVM::P), CvParamGrid nuGrid=get_default_grid(CvSVM::NU), CvParamGridcoeffGrid=get_default_grid(CvSVM::COEF), CvParamGrid degreeGrid=get_default_grid(CvSVM::DEGREE), bool balanced=false )
参数	
k_fold – 交叉验证参数。训练集被分成k_fold的自子集。其中一个子集是用来测试模型，其他子集则成为训练集。所以，SVM算法复杂度是执行k_fold的次数。
*Grid – 对应的SVM迭代网格参数。
balanced – 如果是true则这是一个2类分类问题。这将会创建更多的平衡交叉验证子集。
这个方法根据CvSVMParams中的最佳参数C, gamma, p, nu, coef0, degree自动训练SVM模型。参数被认为是最佳的交叉验证，其测试集预估错误最小。

如果没有需要优化的参数，相应的网格步骤应该被设置为小于或等于1的值。例如，为了避免gamma的优化，设置gamma_grid.step = 0，gamma_grid.min_val， gamma_grid.max_val 为任意数值。所以params.gamma 由gamma得出。

最后，如果参数优化是必需的，但是相应的网格却不确定，你可能需要调用函数CvSVM::get_default_grid()，创建一个网格。例如，对于gamma，调用CvSVM::get_default_grid(CvSVM::GAMMA)。

该函数为分类运行 (params.svm_type=CvSVM::C_SVC 或者 params.svm_type=CvSVM::NU_SVC) 和为回归运行 (params.svm_type=CvSVM::EPS_SVR 或者 params.svm_type=CvSVM::NU_SVR)效果一样好。如果params.svm_type=CvSVM::ONE_CLASS，没有优化，并指定执行一般的SVM。

CvSVM::predict
预测样本的相应数据。

C++: float CvSVM::predict(const Mat& sample, bool returnDFVal=false ) const
C++: float CvSVM::predict(const CvMat* sample, bool returnDFVal=false ) const
C++: float CvSVM::predict(const CvMat* samples, CvMat* results) const
参数	
sample – 需要预测的输入样本。
samples – 需要预测的输入样本们。
returnDFVal – 指定返回值类型。如果值是true，则是一个2类分类问题，该方法返回的决策函数值是边缘的符号距离。
results – 相应的样本输出预测的响应。
这个函数用来预测一个新样本的响应数据(response)。在分类问题中，这个函数返回类别编号；在回归问题中，返回函数值。输入的样本必须与传给trainData的训练样本同样大小。如果训练中使用了varIdx参数，一定记住在predict函数中使用跟训练特征一致的特征。

后缀const是说预测不会影响模型的内部状态，所以这个函数可以很安全地从不同的线程调用。

CvSVM::get_default_grid
生成一个SVM网格参数。

C++: CvParamGrid CvSVM::get_default_grid(int param_id)
参数	
param_id –
SVM参数的IDs必须是下列中的一个：

CvSVM::C
CvSVM::GAMMA
CvSVM::P
CvSVM::NU
CvSVM::COEF
CvSVM::DEGREE
网格参数将根据这个ID生成。

CvSVM::get_params
返回当前SVM的参数。

C++: CvSVMParams CvSVM::get_params() const
这个函数主要是在使用CvSVM::train_auto()时去获得最佳参数。

CvSVM::get_support_vector
检索一定数量的支持向量和特定的向量。

C++: int CvSVM::get_support_vector_count() const
C++: const float* CvSVM::get_support_vector(int i) const
参数	i – 指定支持向量的索引。
该方法可以用于检索一组支持向量。

CvSVM::get_var_count
返回变量的个数。

C++: int CvSVM::get_var_count() const