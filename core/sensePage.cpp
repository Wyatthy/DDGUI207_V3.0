#include "sensePage.h"
#include <QMessageBox>

#include <iostream>
#include <string>
#include <map>
#include <mat.h>

using namespace std;

SenseSetPage::SenseSetPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    projectsInfo(globalProjectInfo)
{
    ui->lineEdit_sense_sampleIndex->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9]\\d*0?[1-9]$|^[1-9]$")));


    // 数据集类别选择框事件相应
    BtnGroup_typeChoice->addButton(ui->radioButton_train_choice, 0);
    BtnGroup_typeChoice->addButton(ui->radioButton_test_choice, 1);
    BtnGroup_typeChoice->addButton(ui->radioButton_val_choice, 2);
    BtnGroup_typeChoice->addButton(ui->radioButton_unknown_choice, 3);

    // connect(this->BtnGroup_typeChoice, &QButtonGroup::buttonClicked, this, &SenseSetPage::changeType);

    // 确定
    connect(ui->pushButton_datasetConfirm, &QPushButton::clicked, this, &SenseSetPage::confirmDataset);

    // 索引取样
    connect(ui->pushButton_sense_sample,&QPushButton::clicked,this,&SenseSetPage::nextBatchChart);

    // 数据集备注保存
    connect(ui->pushButton_saveDatasetNote,&QPushButton::clicked,this,&SenseSetPage::saveDatasetNote);

    // 模型备注保存
    connect(ui->pushButton_saveModelNote,&QPushButton::clicked,this,&SenseSetPage::saveModelNote);


    // 数据集属性显示框
    this->attriLabelGroup["Project_Path"] = ui->label_sense_project;
    this->attriLabelGroup["ProjectType"] = ui->label_sense_datasetType;
    this->attriLabelGroup["datasetClassNum"] = ui->label_sense_claNum;
    this->attriLabelGroup["datasetClassName"] = ui->label_sense_classNames;

    // 模型属性显示
    this->attriLabelGroup["Model_DataType"] = ui->label_sense_modelType;
    this->attriLabelGroup["Model_Algorithm"] = ui->label_sense_modelAlgorithm;
    this->attriLabelGroup["Model_AlgorithmType"] = ui->label_sense_algorithmType;
    this->attriLabelGroup["Model_Framework"] = ui->label_sense_framework;
    this->attriLabelGroup["Model_TrainEpoch"] = ui->label_sense_trainEpoch;
    this->attriLabelGroup["Model_AccuracyOnVal"] = ui->label_sense_valAcc;

    // this->attriLabelGroup["datasetNote"] = ui->lineEdit_datasetNote;

    // 图片显示label成组,一共十个
    imgGroup.push_back(ui->label_datasetClaImg1);
    imgGroup.push_back(ui->label_datasetClaImg1_2);
    imgGroup.push_back(ui->label_datasetClaImg1_3);
    imgGroup.push_back(ui->label_datasetClaImg1_4);
    imgGroup.push_back(ui->label_datasetClaImg1_5);
    imgGroup.push_back(ui->label_datasetClaImg1_6);
    imgGroup.push_back(ui->label_datasetClaImg1_7);
    imgGroup.push_back(ui->label_datasetClaImg1_8);
    imgGroup.push_back(ui->label_datasetClaImg1_9);
    imgGroup.push_back(ui->label_datasetClaImg1_10);

    imgInfoGroup.push_back(ui->label_datasetCla1);
    imgInfoGroup.push_back(ui->label_datasetCla1_2);
    imgInfoGroup.push_back(ui->label_datasetCla1_3);
    imgInfoGroup.push_back(ui->label_datasetCla1_4);
    imgInfoGroup.push_back(ui->label_datasetCla1_5);
    imgInfoGroup.push_back(ui->label_datasetCla1_6);
    imgInfoGroup.push_back(ui->label_datasetCla1_7);
    imgInfoGroup.push_back(ui->label_datasetCla1_8);
    imgInfoGroup.push_back(ui->label_datasetCla1_9);
    imgInfoGroup.push_back(ui->label_datasetCla1_10);

    // // 显示图表成组
    chartGroup.push_back(ui->label_senseChartInfo1);
    chartGroup.push_back(ui->label_senseChartInfo1_2);
    chartGroup.push_back(ui->label_senseChartInfo1_3);
    chartGroup.push_back(ui->label_senseChartInfo1_4);
    chartGroup.push_back(ui->label_senseChartInfo1_5);
    chartGroup.push_back(ui->label_senseChartInfo1_6);
    chartGroup.push_back(ui->label_senseChartInfo1_7);
    chartGroup.push_back(ui->label_senseChartInfo1_8);
    chartGroup.push_back(ui->label_senseChartInfo1_9);
    chartGroup.push_back(ui->label_senseChartInfo1_10);
}

SenseSetPage::~SenseSetPage(){

}

void SenseSetPage::refreshGlobalInfo(){
    ui->label_sense_project->setText(QString::fromStdString(projectsInfo->pathOfSelectedProject));
    ui->label_sense_datasetType->setText(QString::fromStdString(projectsInfo->dataTypeOfSelectedProject));
    // train文件夹路径
    QString projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    QString trainPath = projectPath + "/train";
    // 把train文件夹下所有的文件夹名字作为类别名字，放在一个QString里面
    QString classNames = "";
    QStringList folders = QDir(trainPath).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < folders.size(); i++) {
        QString folderName = folders.at(i);
        // 最后一个不要,
        if (i == folders.size() - 1) classNames += folderName;
        else classNames += folderName + ",";
    }
    // 得到train文件夹下所有文件夹的个数作为类别数量
    int classNum = folders.size();
    ui->label_sense_claNum->setText(QString::number(classNum));
    ui->label_sense_classNames->setText(classNames);
}

void SenseSetPage::saveDatasetNote()
{
    // 保存至内存
    string type = projectsInfo->dataTypeOfSelectedProject;
    string name = projectsInfo->nameOfSelectedProject;
    if(!type.empty() && !name.empty()){
        string customAttriValue = "";
        // 对plainTextEdit组件
        customAttriValue = ui->lineEdit_datasetNote->text().toStdString();
        if(customAttriValue.empty()){
            customAttriValue = "未定义";
        }
        this->projectsInfo->modifyAttri(type, name, "Dataset_Note", customAttriValue);
        // 保存至.xml,并更新
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);

        // 提醒
        QMessageBox::information(NULL, "属性保存提醒", "数据集备注已修改");
        terminal->print("数据集备注："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改已保存");
    }
    else{
        QMessageBox::warning(NULL, "属性保存提醒", "保存失败！");
        terminal->print("数据集备注："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改无效");
    }
}

void SenseSetPage::saveModelNote()
{
    // 保存至内存
    string type = projectsInfo->dataTypeOfSelectedProject;
    string name = projectsInfo->nameOfSelectedProject;
    if(!type.empty() && !name.empty()){
        string customAttriValue = "";
        // 对plainTextEdit组件
        customAttriValue = ui->lineEdit_modelNote->text().toStdString();
        if(customAttriValue.empty()){
            customAttriValue = "未定义";
        }
        this->projectsInfo->modifyAttri(type, name, "Model_Note", customAttriValue);
        // 保存至.xml,并更新
        this->projectsInfo->writeToXML(projectsInfo->defaultXmlPath);

        // 提醒
        QMessageBox::information(NULL, "属性保存提醒", "模型备注已修改");
        terminal->print("模型备注："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改已保存");
    }
    else{
        QMessageBox::warning(NULL, "属性保存提醒", "保存失败！");
        terminal->print("模型备注："+QString::fromStdString(type)+"->"+QString::fromStdString(name)+"->属性修改无效");
    }
}

void SenseSetPage::confirmDataset(bool notDialog = false){
    updateAttriLabel();
    QString project_path = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    qDebug() << "project_path: " << project_path;
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("_")[1];
    qDebug() << "selectedType: " << selectedType;
    if(selectedType.isEmpty()||projectsInfo->pathOfSelectedProject==""){
        QMessageBox::warning(NULL, "数据集切换提醒", "数据集切换失败，活动工程或数据集未指定");
        return;
    }
    QString dataset_path;
    if(selectedType == "unknown") dataset_path = project_path + "/" + selectedType + "_test";
    else dataset_path = project_path + "/" + selectedType;
    qDebug() << "dataset_path: " << dataset_path;
    bool ifDbExists = std::filesystem::exists(std::filesystem::u8path(dataset_path.toStdString()));
    if(!ifDbExists){
        QMessageBox::warning(NULL, "数据集切换提醒", "数据集切换失败，该工程下不存在"+selectedType+"数据集");
        return;
    }
    terminal->print("Selected Type: " + selectedType);

    projectsInfo->typeOfSelectedDataset = selectedType;
    projectsInfo->pathOfSelectedDataset = dataset_path.toStdString();
    projectsInfo->nameOfSelectedDataset = project_path.split('/').last().toStdString() + "/" + selectedType.toStdString();


    // 更新classNamesOfSelectedDataset
    // vector<string> subDirNames;
    // dirTools->getDirsplus(subDirNames, projectsInfo->pathOfSelectedDataset);
    // 排除特殊的文件夹
    // auto temp=std::find(subDirNames.begin(),subDirNames.end(),"model_saving");
    // if(temp!=subDirNames.end()) subDirNames.erase(temp);
    // projectsInfo->classNamesOfSelectedDataset = subDirNames;

    projectsInfo->classNamesOfSelectedDataset.clear();
    QStringList folders = QDir(QString::fromStdString(projectsInfo->pathOfSelectedDataset)).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (int i = 0; i < folders.size(); i++) {
        QString folderName = folders.at(i);
        if (folderName.contains("DT")) {
            projectsInfo->classNamesOfSelectedDataset.insert(projectsInfo->classNamesOfSelectedDataset.begin(), folderName.toStdString());
        } else {
            projectsInfo->classNamesOfSelectedDataset.push_back(folderName.toStdString());
        }
    }
    // qDebug()<<"SenseSetPage::confirmDataset dataset_path== "<<dataset_path;
    // 更新属性显示标签

    updateAttriLabel();
    // 图片路径是工程文件夹路径加上工程文件夹名字+.png
    QString imgPath = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/" + projectsInfo->nameOfSelectedProject + ".png");
    // terminal->print("模型图像地址:"+imgPath);
    //ui->label_modelImg->setPixmap(QPixmap(imgPath).scaled(QSize(400,400), Qt::KeepAspectRatio));
    if(std::filesystem::exists(std::filesystem::u8path(imgPath.toStdString()))){
        // qDebug()<<"add....";
        recvShowPicSignal(QPixmap(imgPath), ui->graphicsView_sense_modelImg);
    }else{
        terminal->print("模型图像地址:"+imgPath+"不存在！");
    }

    //搜索最大索引和样本数量

    int maxIndex = 1000000;
    minMatNum(maxIndex);
    if (maxIndex > 0)
        ui->label_sense_allIndex->setText(QString::fromStdString(to_string(maxIndex-1)));
    qDebug() << "maxIndex: " << maxIndex-1;
    QIntValidator *validator = new QIntValidator(ui->lineEdit_sense_sampleIndex);
    validator->setBottom(1);
    validator->setTop(maxIndex-1);
    ui->lineEdit_sense_sampleIndex->setValidator(validator);

    // 绘制类别图
    for(int i = 0; i<folders.size(); i++){
        imgGroup[i]->clear();
        imgInfoGroup[i]->clear();
    }
    drawClassImage();

    // 绘制曲线
    for(int i=0;i<folders.size();i++){
        if (chartGroup[i]->layout()) {
            QLayoutItem* item;
            while ((item = chartGroup[i]->layout()->takeAt(0))) {
                QWidget* widget = item->widget();
                if (widget) {
                    widget->setParent(nullptr);
                    delete widget;
                }
                delete item;
            }
        }
    //    if (chartGroup[i]->pixmap() != nullptr) {
    //        delete chartGroup[i]->pixmap();
    //    }
        chartGroup[i]->clear();
    }
    nextBatchChart();

    // 绘制表格 TODO
    if(!notDialog)
        QMessageBox::information(NULL, "数据集切换提醒", "已成功切换数据集为->"+selectedType);
}


void SenseSetPage::updateAttriLabel(){
    map<string,string> attriContents = projectsInfo->getAllAttri(
        projectsInfo->dataTypeOfSelectedProject,
        projectsInfo->nameOfSelectedProject
    );
    for(auto &currAttriWidget: this->attriLabelGroup){
        currAttriWidget.second->setText(QString::fromStdString(attriContents[currAttriWidget.first]));
    }
   ui->lineEdit_datasetNote->setText(QString::fromStdString(attriContents["datasetNote"]));
   ui->lineEdit_modelNote->setText(QString::fromStdString(attriContents["modelNote"]));
}


void SenseSetPage::drawClassImage(){
    string rootPath = projectsInfo->pathOfSelectedDataset;
    vector<string> subDirNames = projectsInfo->classNamesOfSelectedDataset;
    // 打印类别名称
    // for(int i=0; i<subDirNames.size(); i++){
    //     qDebug() << "subDirNames[" << i << "]: " << QString::fromStdString(subDirNames[i]);
    // }
    // 显示图片大小设置成一样
    int imgWidth = 100;
    int imgHeight = 100;
    // 按类别显示
        for(int i = 0; i<subDirNames.size(); i++){
        imgInfoGroup[i]->setAlignment(Qt::AlignCenter);
        imgInfoGroup[i]->setText(QString::fromStdString(subDirNames[i]));
        QString imgPath = QString::fromStdString(rootPath +"/"+ subDirNames[i] +".png");
        // 图片都显示在同样大小的框里
        imgGroup[i]->setAlignment(Qt::AlignCenter);
        imgGroup[i]->setPixmap(QPixmap(imgPath).scaled(QSize(200,100), Qt::KeepAspectRatio));
        // 把label框的大小设置为一样
        imgGroup[i]->setFixedSize(QSize(200, 100));
    }
}

// 如果传入的参数是""，那么就随机取索引显示图片，如果不是空，则转为int，用这个作为索引
void SenseSetPage::nextBatchChart(){
    QString exampleIdx = ui->lineEdit_sense_sampleIndex->text();
    string rootPath = projectsInfo->pathOfSelectedDataset;
    vector<string> subDirNames = projectsInfo->classNamesOfSelectedDataset;
    // qDebug()<<"(SenseSetPage::nextBatchImage) subDirNames.size()="<<subDirNames.size();
    // 按类别显示
    for(int i=0; i<subDirNames.size(); i++){
        srand((unsigned)time(NULL));
        // 选取类别
        string choicedClass = subDirNames[i];
        string classPath = rootPath +"/"+ choicedClass;
        Chart *previewChart;

        // 选取Mat
        vector<string> allMatFile;
        if(dirTools->getFilesplus(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            //下面这部分代码都是为了让randomIdx在合理的范围内
            int randomIdx = 0;
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空!!!!";return;}
            pMxArray = matGetNextVariable(pMatFile, NULL);
            if(!pMxArray){
                qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!!";
                return;
            }
            int N = mxGetN(pMxArray);  //N 列数
            if(exampleIdx==""){
                randomIdx = N-(rand())%N;
            }else{
                randomIdx = exampleIdx.toInt();
            }
            if(randomIdx > 0 && randomIdx <= N){
                previewChart = new Chart(ui->label_mE_chartGT,QString::fromStdString(projectsInfo->dataTypeOfSelectedProject),matFilePath);
                previewChart->drawImage(chartGroup[i],randomIdx);
            }else{
                QMessageBox::information(NULL, "错误", "索引超出范围");
                return;
            }
        }
    }
}


// 所有样本类别中的最小样本数，作为索引最大值
void SenseSetPage::minMatNum(int &minNum)
{
    string rootPath = projectsInfo->pathOfSelectedDataset;
    vector<string> subDirNames = projectsInfo->classNamesOfSelectedDataset;
    for(int i=0; i<subDirNames.size(); i++){
        string choicedClass = subDirNames[i];
        string classPath = rootPath +"/"+ choicedClass;
        vector<string> allMatFile;
        if(dirTools->getFilesplus(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            //下面这部分代码都是为了让randomIdx在合理的范围内（
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空!!!!";return;}
            pMxArray = matGetNextVariable(pMatFile, NULL);
            if(!pMxArray){
                qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!!!!";
                return;
            }
            int windowlen = 16;
            int windowstep = 1;
            int N = mxGetN(pMxArray);  //N 列数
            if(projectsInfo->dataTypeOfSelectedProject == "RCS" || projectsInfo->dataTypeOfSelectedProject == "IMAGE"){
                N = (N-windowlen)/windowstep+1;
            }
            if(N<minNum) minNum = N;
        }
    }
}


void SenseSetPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.9;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.9;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}
