#include "modelEvalPage.h"
#include <QMessageBox>
#include <QGraphicsScene>
#include <QChart>
#include <QBarSeries>
#include <QBarSet>
#include <QBarCategoryAxis>
#include <thread>
#include "./lib/guiLogic/tools/guithreadrun.h"
#include<cuda_runtime.h>

#include<Windows.h>  //for Sleep func
using namespace std;

ModelEvalPage::ModelEvalPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{
    GuiThreadRun::inst();
    // 下拉框
    connect(ui->comboBox_sampleType, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_sampleType(QString)));
    connect(ui->comboBox_chosFile, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_chosFile(QString)));
    // 取样按钮
    connect(ui->pushButton_mE_randone, &QPushButton::clicked, this, &ModelEvalPage::takeSample);
    // 测试按钮
    connect(ui->pushButton_testOneSample, &QPushButton::clicked, this, &ModelEvalPage::testOneSample);
    connect(ui->pushButton_testAllSample, &QPushButton::clicked, this, &ModelEvalPage::testAllSample);

    // 多线程的信号槽绑定
    processDatasetInfer = new QProcess();
    connect(processDatasetInfer, &QProcess::readyReadStandardOutput, this, &ModelEvalPage::processDatasetInferFinished);
    processSampleInfer = new QProcess();
    connect(processSampleInfer, &QProcess::readyReadStandardOutput, this, &ModelEvalPage::processSampleInferFinished);

    //cmd调用python做优化模型的推理
    this->condaEnvName = "PT";
    this->pythonApiPath = "./lib/algorithm/optimizeInfer/optimizeInfer.py";

    //混淆矩阵模块的py嵌入
    Py_SetPythonHome(L"D:/win_anaconda");
    Py_Initialize();
    _import_array();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./lib/guiLogic/tools/')");
    pModule_drawConfusionMatrix = PyImport_ImportModule("EvalPageConfusionMatrix");
    pFunc_drawConfusionMatrix = PyObject_GetAttrString(pModule_drawConfusionMatrix, "draw_confusion_matrix");

}

ModelEvalPage::~ModelEvalPage(){

}

void ModelEvalPage::on_comboBox_sampleType(QString choicedClassq){
    this->choicedClass = choicedClassq.toStdString();
    this->choicedFileInClass = "NULL";
    string datasetPath = projectsInfo->pathOfSelectedDataset;
    datasetPath += "/"+choicedClass;
    QStringList files;
    QStringList filters;
    filters << "*.mat"; 
    files = QDir(QString::fromStdString(datasetPath)).entryList(filters, QDir::Files);
    ui->comboBox_chosFile->clear();
    ui->comboBox_chosFile->addItems(files);
}

void ModelEvalPage::on_comboBox_chosFile(QString choicedFileName){
    this->choicedFileInClass = choicedFileName.toStdString();
    this->choicedSamplePATH = 
        projectsInfo->pathOfSelectedDataset + "/" + this->choicedClass + "/" + this->choicedFileInClass;
}

void ModelEvalPage::refreshGlobalInfo(){
    label2class.clear();
    class2label.clear();
    // 单样本测试下拉框刷新
    vector<string> comboBoxContents = projectsInfo->classNamesOfSelectedDataset;
    ui->comboBox_sampleType->clear();
    for(auto &item: comboBoxContents){
        ui->comboBox_sampleType->addItem(QString::fromStdString(item));
    }
    ui->comboBox_inferBatchsize->clear();
    // for(int i=512;i>3;i/=2){
    //     ui->comboBox_inferBatchsize->addItem(QString::number(i));
    // }
    ui->comboBox_inferBatchsize->addItem(QString::number(1));
    ui->comboBox_inferBatchsize->addItem(QString::number(16));
    ui->comboBox_inferBatchsize->addItem(QString::number(32));
    ui->comboBox_inferBatchsize->addItem(QString::number(64));
    ui->comboBox_inferBatchsize->addItem(QString::number(100));
    // 网络输出标签对应类别名称初始化
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++)   label2class[i]=comboBoxContents[i];
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    // 基本信息更新
    ui->label_mE_dataset->setText(QString::fromStdString(projectsInfo->nameOfSelectedDataset));
    ui->label_mE_model->setText(QString::fromStdString(projectsInfo->nameOfSelectedModel_forInfer));
    //ui->label_mE_batch->setText(QString::fromStdString(modelInfo->getAttri(modelInfo->selectedType, modelInfo->selectedName, "batch")));
    if((projectsInfo->pathOfSelectedModel_forInfer!=choicedModelPATH) ||
       (projectsInfo->pathOfSelectedDataset!=choicedDatasetPATH)){//保证模型切换后trt对象重新构建
        trtInfer = new TrtInfer(class2label);
        choicedDatasetPATH = projectsInfo->pathOfSelectedDataset;
        choicedModelPATH = projectsInfo->pathOfSelectedModel_forInfer;
        //TODO 
        this->choicedClass = "NULL";
        this->choicedFileInClass = "NULL";
        ui->comboBox_chosFile->clear();
        // ui->comboBox_sampleType->clear();
    }
}

void ModelEvalPage::takeSample(){
    Chart *previewChart;
    string nameOfMatFile = this->choicedFileInClass;

    QString matFilePath = QString::fromStdString(this->choicedSamplePATH);
    // std::filesystem::exists(this->choicedSamplePATH)
    qDebug()<<"ModelEvalPage::takeSample matFilePath = "<<matFilePath;
    
    // if(dirTools->exist(this->choicedSamplePATH)){
    if(std::filesystem::exists(std::filesystem::u8path(this->choicedSamplePATH))){
        QString examIdx_str = ui->lineEdit_evalSampleIdx->text();
        int examIdx = 1;
        if(examIdx_str==""){
            examIdx=1;
            ui->lineEdit_evalSampleIdx->setText("1");
        }
        else examIdx = examIdx_str.toInt();

        QString imgPath = QString::fromStdString(choicedDatasetPATH +"/"+ this->choicedClass +".png");
        //下面这部分代码都是为了让randomIdx在合理的范围内（
        MATFile* pMatFile = NULL;
        mxArray* pMxArray = NULL;
        pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
        if(!pMatFile){qDebug()<<"(ModelEvnameOfMatFilealPage::takeSample)文件指针空！！！！！！";return;}
        std::string matVariable=nameOfMatFile.substr(0,nameOfMatFile.find_last_of('.')).c_str();//假设数据变量名同文件名的话

        QString chartTitle="Temporary Title";
        if(projectsInfo->dataTypeOfSelectedProject=="HRRP") {chartTitle="HRRP(Ephi),Polarization HP(1)[Magnitude in dB]";}// matVariable="hrrp128";}
        else if (projectsInfo->dataTypeOfSelectedProject=="RADIO") {chartTitle="RADIO Temporary Title";}// matVariable="radio101";}

        pMxArray = matGetVariable(pMatFile,matVariable.c_str());
        if(!pMxArray){qDebug()<<"(ModelEvalPage::takeSample)pMxArray变量没找到！！！！！！";return;}
        int N = mxGetN(pMxArray);  //N 列数
        examIdx = examIdx>N?N-1:examIdx;

        this->emIndex=examIdx;
        // 可视化所选样本
        // ui->label_mE_choicedSample->setText("Index:"+QString::number(randomIdx));
        ui->label_mE_imgGT->setPixmap(QPixmap(imgPath).scaled(QSize(100,100), Qt::KeepAspectRatio));
        //绘图
        previewChart = new Chart(ui->label_mE_chartGT,QString::fromStdString(projectsInfo->dataTypeOfSelectedProject),matFilePath);
        previewChart->drawImage(ui->label_mE_chartGT,examIdx);
    }
    
    else{
        QMessageBox::warning(NULL, "数据取样", "数据取样失败，未指定数据样本或其不存在!");
    }


}

// 移除布局子控件
void removeLayout(QLayout *layout){
    QLayoutItem *child;
    if (layout == nullptr)
        return;
    while ((child = layout->takeAt(0)) != nullptr){
        // child可能为QLayoutWidget、QLayout、QSpaceItem
        // QLayout、QSpaceItem直接删除、QLayoutWidget需删除内部widget和自身
        if (QWidget* widget = child->widget()){
            widget->setParent(nullptr);
            delete widget;
            widget = nullptr;
        }
        else if (QLayout* childLayout = child->layout())
            removeLayout(childLayout);
        delete child;
        child = nullptr;
    }
}

void  ModelEvalPage::testOneSample(){
    struct stat buffer; 
    int modelfileExist=(stat (choicedModelPATH.c_str(), &buffer) == 0);
    QString modelFormat=QString::fromStdString(choicedModelPATH).split('.').last();
    if(!choicedModelPATH.empty() && !choicedSamplePATH.empty()&& (modelfileExist==1)){
        std::cout<<"(ModelEvalPage::testOneSample)choicedSamplePATH"<<choicedSamplePATH<<endl;
        std::vector<float> degrees; int predIdx;
        //classnum==(datasetInfo->selectedClassNames.size())
        std::cout<<"(ModelEvalPage::testOneSample)projectsInfo->dataTypeOfSelectedProject="<<projectsInfo->dataTypeOfSelectedProject<<endl;//HRRP
        std::cout<<"(ModelEvalPage::testOneSample)projectsInfo->modelTypeOfSelectedProject="<<projectsInfo->modelTypeOfSelectedProject<<endl;//TRA_DL
        bool dataProcess=true;std::string flag="";
        if(projectsInfo->dataTypeOfSelectedProject=="HRRP" && projectsInfo->modelTypeOfSelectedProject=="FEA_RELE"){
            QMessageBox::warning(NULL, "提示", "特征关联模型输入应为特征数据");
            return;
        }
        if(projectsInfo->dataTypeOfSelectedProject=="RCS") {
            dataProcess=false;
            flag="RCS_";
        }
        if(projectsInfo->modelTypeOfSelectedProject=="FEA_OPTI"){   
            if(modelFormat!="pth"){
                QMessageBox::warning(NULL, "提示", "优化模型文件需为.pth模型!");
                return;
            }
            // 激活conda python环境
            QString activateEnv = "conda activate "+this->condaEnvName+"&&";
            QString command = activateEnv + "python " + this->pythonApiPath+ \
                " --choicedModelPATH="          + QString::fromStdString(choicedModelPATH)+ \
                " --choicedMatPATH="            + QString::fromStdString(choicedSamplePATH)+ \
                " --choicedSampleIndex="        + QString::number(this->emIndex)+ \
                " --inferMode=sample";

            // 执行python脚本
            this->terminal->print(command);
            this->execuCmdProcess(processSampleInfer,command);
            return;
        }
        if(projectsInfo->modelTypeOfSelectedProject=="INCRE") dataProcess=false; //目前的增量模型接受的数据是没做预处理的
        if(projectsInfo->modelTypeOfSelectedProject=="ABFC"){
            std::string feaWeightTxtPath=choicedModelPATH.substr(0, choicedModelPATH.rfind("/"))+"/attention.txt";
            if(this->dirTools->exist(feaWeightTxtPath)){//abfc有attention文件
                int modelIdx=1,tempi=0;std::vector<int> dataOrder;std::string line;
                std::ifstream infile(feaWeightTxtPath);
                while (getline(infile, line)){
                    if(++tempi==40){modelIdx=std::stoi(line);break;}
                    dataOrder.push_back(std::stoi(line));
                }infile.close();
                trtInfer->setParmsOfABFC(modelIdx, dataOrder);
            }
        }
        if(projectsInfo->modelTypeOfSelectedProject=="ATEC"){
            dataProcess=false;
        }
        if(modelFormat!="trt"){
            QMessageBox::warning(NULL, "提示", "模型文件需为.trt模型!");
            return;
        }
        QString inferTime=trtInfer->testOneSample(choicedSamplePATH, this->emIndex, choicedModelPATH, dataProcess , &predIdx, degrees, flag);
        ui->label_predTime->setText(inferTime);
        /*************************把下面都当做对UI的操作***************************/
        QString predClass = QString::fromStdString(label2class[predIdx]);   // 预测类别

        // 可视化结果
        ui->label_predClass->setText(predClass);
        ui->label_predDegree->setText(QString("%1").arg(degrees[predIdx]*100));
        QString imgPath = QString::fromStdString(choicedDatasetPATH) +"/"+ predClass +".png";
        ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
        std::cout<<"(ModelEvalPage::testOneSample)degrees:";
        for(int i=0;i<degrees.size();i++){
            std::cout<<degrees[i]<<" ";
            degrees[i]=round(degrees[i] * 100) / 100;
        }
        // 绘制隶属度柱状图
        disDegreeChart(predClass, degrees, label2class);

    }
    else{
        QMessageBox::warning(NULL, "单样本测试", "数据或模型未指定！(检查模型路径是否存在)");
    }
}

void ModelEvalPage::testAllSample(){
    // 获取批处理量
    int inferBatch = ui->comboBox_inferBatchsize->currentText().toInt();
    QString modelFormat=QString::fromStdString(choicedModelPATH).split('.').last();
    qDebug()<<"(ModelEvalPage::testAllSample)batchsize=="<<inferBatch;
    /*这里涉及到的全局变量有除了模型数据集路径，还有准确度和混淆矩阵*/
    if(!choicedDatasetPATH.empty() && !choicedModelPATH.empty() ){
        qDebug()<<"(ModelEvalPage::testAllSample)choicedDatasetPATH=="<<QString::fromStdString(choicedDatasetPATH);
        qDebug()<<"(ModelEvalPage::testAllSample)choicedModelPATH=="<<QString::fromStdString(choicedModelPATH);
        // QMessageBox::warning(NULL, "提示", "就先止步于此叭~");
        // return;
        qDebug()<<"(ModelEvalPage::testAllSample)dataTypeOfSelectedProject=="<<QString::fromStdString(projectsInfo->dataTypeOfSelectedProject);
        qDebug()<<"(ModelEvalPage::testAllSample)modelTypeOfSelectedProject=="<<QString::fromStdString(projectsInfo->modelTypeOfSelectedProject);

        float acc = 0.96;
        int classNum=label2class.size();
        std::vector<std::vector<int>> confusion_matrix(classNum, std::vector<int>(classNum, 0));
        std::vector<std::vector<float>> degrees_matrix(classNum, std::vector<float>(0, 0));
        // std::vector<std::vector<float>> degrees_matrix;
        bool dataProcess=true;
        std::string flag="";
        //下面判断一些数据处理的情况
        if(projectsInfo->dataTypeOfSelectedProject=="RCS") {
            dataProcess=false;
            flag="RCS_";
        }
        else if(projectsInfo->dataTypeOfSelectedProject=="FEATURE") {
            flag="ABFC";
        }
        else if(projectsInfo->dataTypeOfSelectedProject=="ATEC") {
            dataProcess=false;
        }
        else if(projectsInfo->modelTypeOfSelectedProject=="CIL") dataProcess=false; //目前增量模型接受的数据是不做预处理的
        if(projectsInfo->modelTypeOfSelectedProject=="ABFC"){
            std::string feaWeightTxtPath=choicedModelPATH.substr(0, choicedModelPATH.rfind("/"))+"/attention.txt";
            if(this->dirTools->exist(feaWeightTxtPath)){//判断是abfc还是atec,依据就是有没有attention文件
                int modelIdx=1,tempi=0;std::vector<int> dataOrder;std::string line;
                std::ifstream infile(feaWeightTxtPath);
                while (getline(infile, line)){//找到目标模型索引和特征顺序
                    if(++tempi==40){modelIdx=std::stoi(line);break;}
                    dataOrder.push_back(std::stoi(line));
                    //cout<<std::stoi(line)<<endl;
                }infile.close();
                trtInfer->setParmsOfABFC(modelIdx, dataOrder);
                flag="ABFC";
            }
        }
        if(projectsInfo->modelTypeOfSelectedProject=="FEA_OPTI"){
            if(modelFormat!="pth"){
                QMessageBox::warning(NULL, "提示", "优化模型文件需为.pth模型!");
                return;
            }
            // 激活conda python环境
            QString activateEnv = "conda activate "+this->condaEnvName+"&&";
            QString command = activateEnv + "python " + this->pythonApiPath+ \
                " --choicedDatasetPATH="        + QString::fromStdString(choicedDatasetPATH)+ \
                " --choicedModelPATH="          + QString::fromStdString(choicedModelPATH)+ \
                " --inferMode=dataset";
            // 执行python脚本
            this->terminal->print(command);
            this->execuCmdProcess(processDatasetInfer, command);
            return;
        }
        if(modelFormat!="trt"){
            QMessageBox::warning(NULL, "提示", "模型文件需为.trt模型!");
            return;
        }
        if(!trtInfer->testAllSample(choicedDatasetPATH,choicedModelPATH,inferBatch,dataProcess,acc,confusion_matrix,flag,degrees_matrix)){
            qDebug()<<"(modelEvalPage::testAllSample) trtInfer-testAll failed~";
            return ;
        }
        
        /*************************Use Python Draw Confusion Matrix******************************/
        int* numpyptr= new int[classNum*classNum];
        for(int i=0;i<classNum;i++){
            for(int j=0;j<classNum;j++){
                numpyptr[i*classNum+j]=confusion_matrix[i][j];
            }
        }

        npy_intp dims[2] = {classNum,classNum};//矩阵维度
        PyArray = PyArray_SimpleNewFromData(2, dims, NPY_INT, numpyptr);//将数据变为numpy
        //用tuple装起来传入python
        args_draw = PyTuple_New(2);
        std::string stringparm="";
        for(int i=0;i<classNum;i++) stringparm=stringparm+label2class[i]+"#";
        PyTuple_SetItem(args_draw, 0, Py_BuildValue("s", stringparm.c_str()));
        PyTuple_SetItem(args_draw, 1, PyArray);
        //函数调用
        pRet_draw = (PyArrayObject*)PyObject_CallObject(pFunc_drawConfusionMatrix, args_draw);
        delete [ ] numpyptr;
        qDebug()<<"(ModelEvalPage::testAllSample) python done";
        /*************************Draw Done******************************/

        //显示混淆矩阵到前端
        QString imgPath = QString::fromStdString("./confusion_matrix.jpg");
        if(all_Images[ui->graphicsView_3_evalpageMatrix]){ //delete 原来的图
            qgraphicsScene->removeItem(all_Images[ui->graphicsView_3_evalpageMatrix]);
            delete all_Images[ui->graphicsView_3_evalpageMatrix]; //空悬指针
            all_Images[ui->graphicsView_3_evalpageMatrix]=NULL;
        }
        if(this->dirTools->exist(imgPath.toStdString())){
            recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_3_evalpageMatrix);
        }
        ui->label_testAllAcc->setText(QString("%1").arg(acc*100));

        for(int i=0;i<label2class.size();i++){
            QLabel *imageLabel=new QLabel("数据集样本在"+QString::fromStdString(label2class[i])+"类上的隶属度曲线");
            QLabel *imageLabel_sig=new QLabel();
            imageLabel_sig->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred); 
            imageLabel_sig->setStyleSheet("border: 3px black");
            QVector<float> meaninglessCiQ = QVector<float>(degrees_matrix[i].begin(), degrees_matrix[i].end());
            Chart *previewChart = new Chart(imageLabel_sig,"usualData","");
            previewChart->drawImageWithSingleSignal(imageLabel_sig,meaninglessCiQ);
            imageLabel_sig->setMinimumHeight(120);
            ui->verticalLayout_22->addWidget(imageLabel);
            ui->verticalLayout_22->addWidget(imageLabel_sig);
        }
        
        // qDebug()<<"ModelEvalPage::testall degrees_matrix.size()"<<degrees_matrix.size()<<" degrees_matrix[0].size()=="<<degrees_matrix[0].size();

        QMessageBox::information(NULL, "所有样本测试", "识别结果已输出！");

    }
    else{
        QMessageBox::warning(NULL, "所有样本测试", "数据集或模型未指定！");
    }
}

void ModelEvalPage::disDegreeChart(QString &classGT, std::vector<float> &degrees, std::map<int, std::string> &classNames){

    QChart *chart = new QChart;
    //qDebug() << "(ModelEvalPage::disDegreeChart)子线程id：" << QThread::currentThreadId();
    std::map<QString, vector<float>> mapnum;
    mapnum.insert(pair<QString, vector<float>>(classGT, degrees));  //后续可拓展
    QBarSeries *series = new QBarSeries();
    map<QString, vector<float>>::iterator it = mapnum.begin();
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
    chart->setAxisX(axis, series);
    chart->setAxisY(axisy, series);
    chart->legend()->setVisible(true);

    QChartView *view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);
    removeLayout(ui->horizontalLayout_degreeChart);
    ui->horizontalLayout_degreeChart->addWidget(view);
    QMessageBox::information(NULL, "单样本测试", "识别成果，结果已输出！");
}

void ModelEvalPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    //QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}

void ModelEvalPage::execuCmdProcess(QProcess *processInfer, QString cmd){
    if(processInfer->state()==QProcess::Running){
        processInfer->close();
        processInfer->kill();
    }
    processInfer->setProcessChannelMode(QProcess::MergedChannels);
    processInfer->start("cmd.exe");
    processInfer->write(cmd.toLocal8Bit() + '\n');
}

void ModelEvalPage::processDatasetInferFinished(){
    float valAcc_optim = 66.6;
    QByteArray cmdOut = processDatasetInfer->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            if(processDatasetInfer->state()==QProcess::Running){
                processDatasetInfer->close();
                processDatasetInfer->kill();
            }
            QStringList loglist = logs.split("$");
            if(loglist.length()!=0)
                ui->label_testAllAcc->setText(loglist[1]);
            QMessageBox::information(NULL, "所有样本测试", "识别成果，结果已输出！");

            // 加载图像
            QString cMatrixPath = QString::fromStdString(choicedModelPATH);
            cMatrixPath = cMatrixPath.left(cMatrixPath.lastIndexOf('/'));
            QString imgPath = cMatrixPath + QString::fromStdString("./confusion_matrix_temp.jpg");
            if(all_Images[ui->graphicsView_3_evalpageMatrix]){ //delete 原来的图
                qgraphicsScene->removeItem(all_Images[ui->graphicsView_3_evalpageMatrix]);
                delete all_Images[ui->graphicsView_3_evalpageMatrix]; //空悬指针
                all_Images[ui->graphicsView_3_evalpageMatrix]=NULL;
            }
            if(this->dirTools->exist(imgPath.toStdString())){
                recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_3_evalpageMatrix);
            }

            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) Logs:"<<logs;
        }
        if(logs.contains("Error") || logs.contains("Errno")){
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) 优化模型推理失败";
            terminal->print("优化模型推理失败");
            QMessageBox::warning(NULL,"错误","something wrong!");
            qDebug()<<"(ModelEvalPage::processDatasetInferFinished) Logs:"<<logs;
        }
    }
}

void ModelEvalPage::processSampleInferFinished(){
    std::vector<float> degrees; int predIdx;
    QByteArray cmdOut = processSampleInfer->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        if(logs.contains("finished")){
            if(processSampleInfer->state()==QProcess::Running){
                processSampleInfer->close();
                processSampleInfer->kill();
            }
            //提取log里的degrees和predIdx信息
            QStringList loglist;
            loglist = logs.split("$");
            if(loglist.length() == 0){
                QMessageBox::warning(NULL,"错误","模型脚本无输出!");
                return;
            }
            for(int i=0;i<loglist.length()-3;i++){   //[-1:predIdx\inferCost\dump]
                degrees.push_back(loglist[i].toFloat());
            }
            predIdx=loglist[loglist.length()-3].toInt();
            ui->label_predTime->setText(loglist[loglist.length()-2]);

            trtInfer->softmax(degrees);

            /*************************把下面都当做对UI的操作***************************/
            QString predClass = QString::fromStdString(label2class[predIdx]);   // 预测类别

            // 可视化结果
            ui->label_predClass->setText(predClass);
            ui->label_predDegree->setText(QString("%1").arg(degrees[predIdx]*100));
            QString imgPath = QString::fromStdString(choicedDatasetPATH) +"/"+ predClass +".png";
            ui->label_predImg->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
            std::cout<<"(ModelEvalPage::testOneSample)degrees:";
            for(int i=0;i<degrees.size();i++){
                std::cout<<degrees[i]<<" ";
                degrees[i]=round(degrees[i] * 100) / 100;
            }
            // 绘制隶属度柱状图
            disDegreeChart(predClass, degrees, label2class);
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) Logs:"<<logs;
        }
        if(logs.contains("Error") || logs.contains("Errno")){
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) 优化模型推理失败";
            terminal->print("优化模型推理失败");
            QMessageBox::warning(NULL,"错误","something wrong!");
            qDebug()<<"(ModelEvalPage::processSampleInferFinished) Logs:"<<logs;
        } 
    }
}
