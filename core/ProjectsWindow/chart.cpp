#include "chart.h"
#include "qapplication.h"
#include "qpushbutton.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <QLegendMarker>
#include <mat.h>
#include <opencv2/opencv.hpp>


Chart::Chart(QWidget* parent, QString dataSetType_, QString _filefullpath){
    setParent(parent);
    dataSetType = dataSetType_;
    if(dataSetType=="HRRP") {chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";}
    else if (dataSetType=="FEATURE") {chartTitle="FEATURE";}
    else if (dataSetType=="RCS") {chartTitle="RCS";}
    else {chartTitle="Temporary Title";}
    filefullpath = _filefullpath;
    series = new QSplineSeries(this);
    qchart = new QChart;
    chartview = new QChartView(qchart);
    layout = new QHBoxLayout(this);
    axisX = new QValueAxis(this);
    axisY = new QValueAxis(this);
    zoom_btn = new QPushButton("放\n大");
    download_btn = new QPushButton("下\n载");

    connect(zoom_btn,&QPushButton::clicked,this,&Chart::ShowBigPic);
    connect(download_btn,&QPushButton::clicked,this,&Chart::SaveBigPic);

    layout->addWidget(chartview);
    layout->setContentsMargins(0,0,0,0);
    setLayout(layout);
    chartview->setRenderHint(QPainter::Antialiasing);//防止图形走样
}


Chart::~Chart(){
    delete qchart;
    delete zoom_btn;
    delete download_btn;
//    qchart=NULL;
//    zoom_btn=NULL;
//    download_btn=NULL;
}

// 将OpenCV的cv::Mat转换为Qt的QImage
QImage matToQImage(const cv::Mat& mat){
    if (mat.type() == CV_8UC1) {
        // 灰度图像
        // return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        cv::Mat mat8bit;
        cv::normalize(mat, mat8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        return QImage(mat8bit.data, mat8bit.cols, mat8bit.rows, static_cast<int>(mat8bit.step), QImage::Format_Grayscale8);
    } else if (mat.type() == CV_8UC3) {
        // RGB彩色图像
        cv::Mat rgbMat;
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
        return QImage(rgbMat.data, rgbMat.cols, rgbMat.rows, static_cast<int>(rgbMat.step), QImage::Format_RGB888);

    } else {
        qWarning() << "Unsupported mat type: " << mat.type();
        return QImage();
    }
}

void Chart::drawHRRPimage(QLabel* chartLabel, int emIdx, int windowlen, int windowstep){
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    // 读取.mat文件（例：mat文件名为"initUrban.mat"，其中包含"initA"）
    double* matdata;
    pMatFile = matOpen(filefullpath.toStdString().c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(MatDataProcess_image:getDataFromMat)文件指针空！！！！！！";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(MatDataProcess:getAllDataFromMat).mat文件变量没找到!!!("<<filefullpath;
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //行数
    int N = mxGetN(pMxArray);  //列数
    int allDataNum=(N-windowlen)/windowstep+1;
    emIdx = emIdx>allDataNum?allDataNum:emIdx;//说明是随机数

    cv::Mat mat(windowlen, M, CV_64FC1);

    for(int i=0;i<windowlen;i++){
        for(int j=0;j<M;j++){
            mat.at<double>(i, j) = matdata[((emIdx-1)*windowstep+i)*M+j];
        }
    }
    cv::Mat mat8bit;
    cv::normalize(mat, mat8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // 调用applyColorMap函数将灰度图转换为热图
    // cv::Mat heatmap;
    // cv::applyColorMap(mat8bit, heatmap, cv::COLORMAP_JET);
    QImage qImage = matToQImage(mat8bit);
    // 将图像缩放以适合QLabel大小
    QPixmap pixmap = QPixmap::fromImage(qImage).scaled(chartLabel->size(), Qt::IgnoreAspectRatio, Qt::FastTransformation);
    // 在QLabel中显示QPixmap
    // chartLabel->setPixmap(pixmap);
    QLabel *label = new QLabel(chartLabel);
    label->setPixmap(pixmap);
    
    QHBoxLayout *pHLayout = (QHBoxLayout *)chartLabel->layout();
    if(!chartLabel->layout()){
        pHLayout = new QHBoxLayout(chartLabel);
    }
    else{
        QLayoutItem *child;
        while ((child = pHLayout->takeAt(0)) != 0){
            if(child->widget()){
                child->widget()->setParent(NULL);
            }
            delete child;
         }
    }
    pHLayout->addWidget(label);
}

void Chart::drawImage(QLabel* chartLabel, int examIdx, int windowlen, int windowstep){
    std::string dataFileFormat=filefullpath.split(".").last().toStdString();
    if(dataSetType=="IMAGE"){
        drawHRRPimage(chartLabel,examIdx, windowlen, windowstep);
        return;
    }
    //把目标样本点添加到points中
    if (dataFileFormat=="mat"&& dataSetType=="HRRP"){
        readHRRPmat(examIdx);
        setAxis("Time/mm",xmin,xmax,10, "dB(V/m)",ymin,ymax,10);
    }else if(dataFileFormat=="mat"&& dataSetType=="FEATURE"){
        readFeaturemat(examIdx);
        setAxis("特征索引",xmin,xmax,10, "特征值",ymin,ymax,10);
        qDebug()<<"read feature matle";
    }else if(dataFileFormat=="mat"&& dataSetType=="RCS"){
        readRCSmat(examIdx, windowlen, windowstep);
        setAxis("Time/mm",xmin,xmax,10, "dB(V/m)",ymin,ymax,10);
    }else return;
    
    //根据数据类型绘制points
    if(dataSetType=="FEATURE"){
        buildChartAsScatter(points);
        showChart(chartLabel);
    }
    else{
        buildChart(points);
        showChart(chartLabel);
    }

}

//绘制已有的数组，在monitorPage回显数据时被调用
void Chart::drawImageWithSingleSignal(QLabel* chartLabel, QVector<float>& dataFrameQ){
    if(dataSetType=="HRRP"){
        points.clear();
        float y_min = 200000,y_max = -200000;
        for(int i=0;i<dataFrameQ.size();i++){
            float y=dataFrameQ[i];
            y_min = fmin(y_min,y);
            y_max = fmax(y_max,y);
            points.append(QPointF(i,y));
        }
        xmin = 0; xmax = dataFrameQ.size()+4;
        ymin = y_min-3; ymax = y_max+3;
        setAxis("Range/cm",xmin,xmax,10, "dB(V/m)",ymin,ymax,10);  
    }else if(dataSetType=="FEATURE"){
        points.clear();
        float y_min = 200000,y_max = -200000;
        for(int i=0;i<dataFrameQ.size();i++){
            float y=dataFrameQ[i];
            y_min = fmin(y_min,y);
            y_max = fmax(y_max,y);
            points.append(QPointF(i,y));
        }
        xmin = 0; xmax = dataFrameQ.size()+4;
        ymin = y_min-3; ymax = y_max+3;
        setAxis("Time/mm",xmin,xmax,10, "dB(V/m)",ymin,ymax,10);
    }
    else if(dataSetType=="usualData"){
        points.clear();
        float y_min = FLT_MAX, y_max = -FLT_MAX;
        for (int i = 0; i < dataFrameQ.size(); i++) {
            float y = dataFrameQ[i];
            y_min = fmin(y_min, y);
            y_max = fmax(y_max, y);
            points.append(QPointF(i, y));
        }
        xmin = -1;
        xmax = dataFrameQ.size() + 1;
        ymin = y_min - 1;
        ymax = y_max + 1;
        setAxis("Sample Index", xmin, xmax, 10, "Degree of sample", ymin, ymax, (ymax - ymin) / 10);
    }
    buildChart(points);
    showChart(chartLabel);
}

void Chart::drawImageWithTwoVector(QLabel* chartLabel, QVector<QVector<float>> dataFrames, QString mesg){
    twoSeries = true ;
    points_mapfea.clear();
    points_tradfea.clear();
    float y_min = FLT_MAX, y_max = -FLT_MAX;
    for (int i = 0; i < dataFrames[0].size(); i++) {
        float y = dataFrames[0][i];
        y_min = fmin(y_min, y);
        y_max = fmax(y_max, y);
        points_mapfea.append(QPointF(i, y));
    }
    for (int i = 0; i < dataFrames[1].size(); i++) {
        float y = dataFrames[1][i];
        y_min = fmin(y_min, y);
        y_max = fmax(y_max, y);
        points_tradfea.append(QPointF(i, y));
    }  
    chartTitle = mesg;
    xmin = 0;
    xmax = dataFrames[0].size() + 4;
    ymin = y_min - 0.2;
    ymax = y_max + 0.2;
    setAxis("Sample Index", xmin, xmax, 10, "Feature Value", ymin, ymax, (ymax - ymin) / 10);
    buildChartWithNiceColor(points_mapfea,points_tradfea);
    showChart(chartLabel);
}

void Chart::readRadiomat(int emIdx){
    points.clear();
    float y_min = 200000,y_max = -200000;
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;

    double* matdata;
    pMatFile = matOpen(filefullpath.toStdString().c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(Chart::readHRRPmat)文件指针空！！！！！！";
        return;
    }
    std::string matVariable=filefullpath.split("/").last().split(".")[0].toStdString().c_str();//假设数据变量名同文件名的话
    //qDebug()<<"(Chart::readRadiomat)matVariable="<<QString::fromStdString(matVariable);
    pMxArray = matGetVariable(pMatFile,matVariable.c_str());
    if(!pMxArray){
        qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到！！！！！！";
        QMessageBox::information(NULL, "绘制错误", "mat文件变量名不合适");
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M=128 行数
    int N = mxGetN(pMxArray);  //N=1000 列数
    if(emIdx>N) emIdx=N-1; //说明是随机数
    for(int i=0;i<M;i++){
        float y=matdata[M*emIdx+i];
        y_min = fmin(y_min,y);
        y_max = fmax(y_max,y);
        points.append(QPointF(i,y));
    }
    //qDebug()<<"(Chart::readHRRPmat)M:"<<M<<"      N:"<<N;
    xmin = 0; xmax = M+4;
    ymin = y_min-3; ymax = y_max+3;
    //qDebug()<<"(Chart::readHRRPmat)ymin:"<<ymin<<"      ymax:"<<ymax;
//    mxFree(pMxArray);
//    matClose(pMatFile);//不注释这两个善后代码就会crashed，可能是冲突了
}

void Chart::readHRRPmat(int emIdx){
    points.clear();
    float y_min = 200000,y_max = -200000;
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;
    double* matdata;
    pMatFile = matOpen(filefullpath.toStdString().c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(Chart::readHRRPmat)文件指针空！！！！！！";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!!!";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M=128 行数
    int N = mxGetN(pMxArray);  //N=1000 列数
    if(emIdx>N) emIdx=N; //说明是随机数
    for(int i=0;i<M;i++){
        float y=matdata[M*(emIdx-1)+i];
        y_min = fmin(y_min,y);
        y_max = fmax(y_max,y);
        points.append(QPointF(i,y));
    }
    xmin = -1; xmax = M+1;
    ymin = y_min-3; ymax = y_max+3;
}

void Chart::readFeaturemat(int emIdx){
    points.clear();
    float y_min = 200000,y_max = -200000;
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;

    double* matdata;
    pMatFile = matOpen(filefullpath.toStdString().c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(Chart::readFeaturemat)文件指针空!!!";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(Chart::readFeaturemat)pMxArray变量没找到!!!";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  //M 行数
    int N = mxGetN(pMxArray);  //N= 列数
    if(emIdx>N) emIdx=N; //说明是随机数
    for(int i=0;i<M;i++){
        float y=matdata[M*(emIdx-1)+i];
        y_min = fmin(y_min,y);
        y_max = fmax(y_max,y);
        points.append(QPointF(i,y));
    }

    xmin = -1; xmax = M+1;
    ymin = y_min-3; ymax = y_max+3;

}

void Chart::readRCSmat(int emIdx, int windowlen, int windowstep){
    points.clear();
    float y_min = 200000,y_max = -200000;
    MATFile* pMatFile = NULL;
    mxArray* pMxArray = NULL;

    double* matdata;
    pMatFile = matOpen(filefullpath.toStdString().c_str(), "r");
    if(!pMatFile){
        qDebug()<<"(Chart::readHRRPmat)文件指针空！！！！！！";
        return;
    }
    pMxArray = matGetNextVariable(pMatFile, NULL);
    if(!pMxArray){
        qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!";
        return;
    }
    matdata = (double*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);  // 行数
    int N = mxGetN(pMxArray);  // 列数
    if(emIdx>(N-windowlen)/windowstep+1) emIdx=(N-windowlen)/windowstep+1;  
    for(int i=0;i<windowlen;i++){
        float y = matdata[(emIdx-1)*windowstep+i];
        y_min = fmin(y_min,y);
        y_max = fmax(y_max,y);
        points.append(QPointF(i,y));
    }
    xmin = -1; xmax = windowlen + 1;
    ymin = y_min-3; ymax = y_max+3;

}

void Chart::readHRRPtxt(){
    float x_min = 200,x_max = -200,y_min = 200,y_max = -200;
    //=======================================================
    //             文件读操作，后续可更换
    //=======================================================
    QFile file(filefullpath);
    //qDebug()<<"(Chart::readHRRPtxt) filefullpath："<<filefullpath;
    if(file.open(QIODevice::ReadOnly)){
        QByteArray line = file.readLine();
        QString str(line);
        points.clear();
        while(!file.atEnd()){
            QByteArray line = file.readLine();
            QString str(line);
            QStringList strList = str.split("\t");
            QStringList result = strList.filter(".");
            if(result.length()==2){
                float x=result[0].toFloat();
                float y=result[1].toFloat();
                points.append(QPointF(x,y));
                x_min = fmin(x_min,x);
                y_min = fmin(y_min,y);
                x_max = fmax(x_max,x);
                y_max = fmax(y_max,y);
            }
        }
        xmin = x_min-3; xmax = x_max+3;
        ymin = y_min-3; ymax = y_max+3;
    }
    else{
        qDebug() << "txt files open filed! ";
    }
}

//调用于trtInfer::realTimeInfer
QWidget* Chart::drawDisDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames){
    QChart *chart = new QChart;
    //qDebug() << "(ModelEvalPage::disDegreeChart)子线程id：" << QThread::currentThreadId();
    std::map<QString, std::vector<float>> mapnum;
    mapnum.insert(std::pair<QString, std::vector<float>>(classGT, degrees));  //后续可拓展
    QBarSeries *series = new QBarSeries();
    std::map<QString, std::vector<float>>::iterator it = mapnum.begin();
    //将数据读入
    while (it != mapnum.end()){
        QString tit = it->first;
        QBarSet *set = new QBarSet(tit);
        std::vector<float> vecnum = it->second;
        for (auto &a : vecnum){
            *set << a;
        }
        series->append(set);
        it++;
    }
    series->setVisible(true);
    series->setLabelsVisible(true);
    // 横坐标参数
    QBarCategoryAxis *axis = new QBarCategoryAxis;
    for(int i = 0; i<classNames.size(); i++){
        axis->append(QString::fromStdString(classNames[i]));
    }
    QValueAxis *axisy = new QValueAxis;
    axisy->setTitleText("隶属度");
    chart->addSeries(series);
    chart->setTitle("识别目标对各类别隶属度分析图");
    //std::cout<<"(ModelEvalPage::disDegreeChart): H444444444444"<<std::endl;
    chart->setAxisX(axis, series);
    chart->setAxisY(axisy, series);
    chart->legend()->setVisible(true);

    QChartView* view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    return view;
}

void Chart::drawImageWithMultipleVector(QLabel* chartLabel, QVector<QVector<float>> dataFrames, QString mesg){
    
    multipleSeries = true ;
    points_list.clear();
    float y_min = FLT_MAX, y_max = -FLT_MAX;
    
    for(int i=0;i<dataFrames.size();i++){
        QList<QPointF> pointsI;
        for (int j = 0; j < dataFrames[i].size(); j++) {
            float y = dataFrames[i][j];
            y_min = fmin(y_min, y);
            y_max = fmax(y_max, y);
            pointsI.append(QPointF(j, y));
        }
        points_list.push_back(pointsI);
    }
    chartTitle = mesg;
    xmin = -1;
    xmax = dataFrames[0].size() + 1;
    ymin = y_min - 0.2;
    ymax = y_max + 0.2;
    setAxis("Sample Index", xmin, xmax, 10, "Degrees", ymin, ymax, (ymax - ymin) / 10);
    buildChartWithMutipleList();
    showChart(chartLabel);
}

void Chart::buildChartWithMutipleList(){
    series_list.clear();
    QColor colors[] = {Qt::blue, Qt::red, Qt::yellow, Qt::black}; 
    // points_list.clear();
    int numOfList = points_list.size();
    for(int i=0;i<numOfList;i++){
        QSplineSeries *seriesI = new QSplineSeries(this);
        seriesI->setPen(QPen(colors[i % 4],0.5,Qt::SolidLine));
        series_list.push_back(seriesI);
        QList<QPointF> pointsI;
        for(int j=0;j<points_list.at(i).size();j++){
            seriesI->append(points_list.at(i).at(j).x(), points_list.at(i).at(j).y());
            // pointsI.append(QPointF(points_list.at(i).at(j).x(), points_list.at(i).at(j).y()));
        }
        qchart->addSeries(seriesI);//输入数据
        qchart->setAxisX(axisX, seriesI);
        qchart->setAxisY(axisY, seriesI);
    }
}

void Chart::setAxis(QString _xname, qreal _xmin, qreal _xmax, int _xtickc, \
             QString _yname, qreal _ymin, qreal _ymax, int _ytickc){
    xname = _xname; xmin = _xmin; xmax = _xmax; xtickc = _xtickc;
    yname = _yname; ymin = _ymin; ymax = _ymax; ytickc = _ytickc;

    axisX->setRange(xmin, xmax);    //设置范围
    axisX->setLabelsVisible(false);   //设置刻度的格式
    axisX->setGridLineVisible(true);   //网格线可见
    axisX->setTickCount(xtickc);       //设置多少个大格
    axisX->setMinorTickCount(1);   //设置每个大格里面小刻度线的数目
    axisX->setTitleText(xname);  //设置描述
    axisX->setTitleVisible(false);
    axisY->setRange(ymin, ymax);
    axisY->setLabelsVisible(false);
    axisY->setGridLineVisible(true);
    axisY->setTickCount(ytickc);
    axisY->setMinorTickCount(1);
    axisY->setTitleText(yname);
    axisY->setTitleVisible(false);
    qchart->addAxis(axisX, Qt::AlignBottom); //下：Qt::AlignBottom  上：Qt::AlignTop
    qchart->addAxis(axisY, Qt::AlignLeft);   //左：Qt::AlignLeft    右：Qt::AlignRight
    qchart->setContentsMargins(-10, -10, -10, -10);  //设置外边界全部为0
    qchart->setMargins(QMargins(-25, 0, -10, -15));
}


void Chart::buildChart(QList<QPointF> pointlist){
    //创建数据源
    series->setPen(QPen(Qt::blue,0.5,Qt::SolidLine));
    series->clear();
    points.clear();
    for(int i=0; i<pointlist.size();i++){
        series->append(pointlist.at(i).x(), pointlist.at(i).y());
        points.append(QPointF(pointlist.at(i).x(), pointlist.at(i).y()));
    }

    qchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式
    qchart->legend()->hide(); //隐藏图例
    qchart->addSeries(series);//输入数据
    qchart->setAxisX(axisX, series);
    qchart->setAxisY(axisY, series);
}

void Chart::buildChartAsScatter(QList<QPointF> pointlist){
    //创建数据源
    scatterSeries = new QScatterSeries(this);
    scatterSeries->setPen(QPen(Qt::blue,0.1));

    scatterSeries->clear();
    points.clear();
    for(int i=0; i<pointlist.size();i++){
        scatterSeries->append(pointlist.at(i).x(), pointlist.at(i).y());
        points.append(QPointF(pointlist.at(i).x(), pointlist.at(i).y()));
    }

    qchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式
    qchart->legend()->hide(); //隐藏图例
    qchart->addSeries(scatterSeries);//输入数据
    qchart->setAxisX(axisX, series);
    qchart->setAxisY(axisY, series);
}

void Chart::buildChartWithNiceColor(QList<QPointF> pointlistF1, QList<QPointF> pointlistF2){
    //创建数据源
    series_mapfea = new QSplineSeries(this);
    series_tradfea = new QSplineSeries(this);
    series_mapfea->setPen(QPen(Qt::red,0.5,Qt::SolidLine));
    series_tradfea->setPen(QPen(Qt::blue,0.5,Qt::SolidLine));
    series_mapfea->clear();
    series_tradfea->clear();
    points_mapfea.clear();
    points_tradfea.clear();


    for(int i=0;i<pointlistF1.size();i++){
        series_mapfea->append(pointlistF1.at(i).x(), pointlistF1.at(i).y());
        points_mapfea.append(QPointF(pointlistF1.at(i).x(), pointlistF1.at(i).y()));
    }
    for(int i=0;i<pointlistF1.size();i++){
        series_tradfea->append(pointlistF2.at(i).x(), pointlistF2.at(i).y());
        points_tradfea.append(QPointF(pointlistF2.at(i).x(), pointlistF2.at(i).y()));
    }



    // qchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式
    qchart->legend()->hide(); //隐藏图例
    qchart->addSeries(series_mapfea);//输入数据
    qchart->addSeries(series_tradfea);//输入数据

    qchart->setAxisX(axisX, series_mapfea);
    qchart->setAxisX(axisX, series_tradfea);
    qchart->setAxisY(axisY, series_mapfea);
    qchart->setAxisY(axisY, series_tradfea);

}

void Chart::showChart(QLabel *imagelabel){
    QHBoxLayout *pHLayout = (QHBoxLayout *)imagelabel->layout();
    if(!imagelabel->layout()){
        pHLayout = new QHBoxLayout(imagelabel);
    }
    else{
        QLayoutItem *child;
        while ((child = pHLayout->takeAt(0)) != 0){
            if(child->widget()){
                child->widget()->setParent(NULL);
            }
            delete child;
         }
    }
    QVBoxLayout *subqvLayout = new QVBoxLayout();

    zoom_btn->setFixedSize(20,35);
    download_btn->setFixedSize(20,35);

    subqvLayout->addWidget(zoom_btn);
    subqvLayout->addWidget(download_btn);
    subqvLayout->setAlignment(Qt::AlignCenter);
    subqvLayout->setContentsMargins(0, 0, 0, 0);

    QWidget* Widget = new QWidget;
    Widget->setLayout(subqvLayout);
    Widget->setContentsMargins(0, 0, 0, 0);

    pHLayout->addWidget(this, 20);
    pHLayout->addWidget(Widget, 1);
    pHLayout->setContentsMargins(0, 0, 0, 0);
}


void Chart::Show_infor(){
    QStringList spilited_names = filefullpath.split('/');
    int id = spilited_names.length()-2;
    QString cls = spilited_names[id];
    QString content = "文件路径:   "+filefullpath+"\n"+"类别标签:   "+cls;
    QMessageBox::about(NULL, "文件信息", content);
}


void Chart::ShowBigPic(){
    ShoworSave = 1;
    Show_Save();
}


void Chart::SaveBigPic(){
    ShoworSave = 2;
    Show_Save();
}


void Chart::Show_Save(){
    QChart *newchart = new QChart();
    QValueAxis *newaxisX = new QValueAxis();
    QValueAxis *newaxisY = new QValueAxis();
    newaxisX->setRange(xmin, xmax);    //设置范围
    newaxisX->setLabelFormat("%d");   //设置刻度的格式
    newaxisX->setGridLineVisible(true);   //网格线可见
    newaxisX->setTickCount(xtickc);       //设置多少个大格
    newaxisX->setMinorTickCount(1);   //设置每个大格里面小刻度线的数目
    newaxisX->setTitleText(xname);  //设置描述
    newaxisX->setTitleVisible(true);
    newaxisY->setRange(ymin, ymax);
    newaxisY->setLabelFormat("%d");
    newaxisY->setGridLineVisible(true);
    newaxisY->setTickCount(ytickc);
    newaxisY->setMinorTickCount(1);
    newaxisY->setTitleText(yname);
    newaxisY->setTitleVisible(true);
    newchart->addAxis(newaxisX, Qt::AlignBottom); //下：Qt::AlignBottom  上：Qt::AlignTop
    newchart->addAxis(newaxisY, Qt::AlignLeft);   //左：Qt::AlignLeft    右：Qt::AlignRight
    newchart->setContentsMargins(0, 0, 0, 0);  //设置外边界全部为0
    newchart->setMargins(QMargins(0, 0, 0, 0));
    // newchart->setAnimationOptions(QChart::SeriesAnimations);//设置曲线动画模式
    if(dataSetType=="FEATURE"){//数据类型如果是特征就绘制散点图
        QScatterSeries *newseries = new QScatterSeries();
        newseries->setPen(QPen(Qt::blue,1));
        newseries->clear();
        for(int i=0; i<points.size();i++)
            newseries->append(points.at(i).x(), points.at(i).y());
        newchart->setTitle(chartTitle);
        newchart->legend()->hide(); //隐藏图例
        newchart->addSeries(newseries);//输入数据
        newchart->setAxisX(newaxisX, newseries);
        newchart->setAxisY(newaxisY, newseries);
    }
    else if(multipleSeries){
        QColor colors[] = {Qt::blue, Qt::red, Qt::yellow, Qt::black}; 
        // QString legendTexts[] = {"Series 1", "Series 2", "Series 3", "Series 4"};  // 定义图例文本数组
        QLegend *legend = newchart->legend();  // 获取图表对象的图例
        for(int i = 0;i<series_list.size();i++){
            QSplineSeries *newseriesI = new QSplineSeries();
            newseriesI->clear();
            newseriesI->setPen(QPen(colors[i % 4], 1, Qt::SolidLine)); 
            for(int j=0; j<points_list[i].size();j++)
                newseriesI->append(points_list[i].at(j).x(), points_list[i].at(j).y());
            newchart->addSeries(newseriesI);//输入数据
            newchart->setAxisX(newaxisX, newseriesI);
            newchart->setAxisY(newaxisY, newseriesI);
            QLegendMarker *marker = legend->markers(newseriesI).first();  // 获取该 series 对应的图例标记
            marker->setVisible(true);  // 显示该图例标记
            marker->setLabel(legendList[i % 4]);  // 使用模运算符取模获取对应的图例文本
        }
        newchart->setTitle(chartTitle);
        
    }else if(twoSeries){
        QSplineSeries *newseriesA = new QSplineSeries();
        newseriesA->setPen(QPen(Qt::red,1,Qt::SolidLine));
        newseriesA->clear();
        QSplineSeries *newseriesB = new QSplineSeries();
        newseriesB->setPen(QPen(Qt::blue,1,Qt::SolidLine));
        newseriesB->clear();
        for(int i=0; i<points_mapfea.size();i++)
            newseriesA->append(points_mapfea.at(i).x(), points_mapfea.at(i).y());
        for(int i=0; i<points_tradfea.size();i++)
            newseriesB->append(points_tradfea.at(i).x(), points_tradfea.at(i).y());
        newchart->setTitle(chartTitle);
        // newchart->legend()->hide(); //隐藏图例
        // QLegendMarker *markerA = newchart->legend()->markers().at(0);
        // markerA->setLabel("mapping_feature");
        // QLegendMarker *markerB = newchart->legend()->markers().at(1);
        // markerB->setLabel("traditional_feature");
        newchart->addSeries(newseriesA);//输入数据
        newchart->addSeries(newseriesB);
        newchart->setAxisX(newaxisX, newseriesA);
        newchart->setAxisY(newaxisY, newseriesA);
        newchart->setAxisX(newaxisX, newseriesB);
        newchart->setAxisY(newaxisY, newseriesB);
    }else{
        QSplineSeries *newseries = new QSplineSeries();
        newseries->setPen(QPen(Qt::blue,1,Qt::SolidLine));
        newseries->clear();
        for(int i=0; i<points.size();i++)
            newseries->append(points.at(i).x(), points.at(i).y());
        newchart->setTitle(chartTitle);
        newchart->legend()->hide(); //隐藏图例
        newchart->addSeries(newseries);//输入数据
        newchart->setAxisX(newaxisX, newseries);
        newchart->setAxisY(newaxisY, newseries);
    }


    newchart->resize(800,600);
    QChartView *bigView = new QChartView(newchart);
    bigView->setRenderHint(QPainter::Antialiasing);

    if(ShoworSave==1){
        bigView->show();
    }
    else{
        QPixmap p = bigView->grab();
        QImage image = p.toImage();
        QString fileName = QFileDialog::getSaveFileName(this,tr("保存文件"),"",tr("chart(*.png)"));
        if(!fileName.isNull()){
            image.save(fileName);
            QFileInfo file(fileName);
            if(file.exists()){
                QMessageBox::about(this, "操作成功", "文件保存成功!");
            }
            else{
                QMessageBox::about(this, "操作失败", "文件保存失败，请重试!");
            }
        }
    }
}

void Chart::setLegend(QStringList legendlist){
    legendList = legendlist;
}
