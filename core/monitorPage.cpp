#include "monitorPage.h"
#include "qimagereader.h"
#include <iostream>
#include <QMessageBox>
#include <QMutex>
#include <QFileInfo>

MonitorPage::MonitorPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo,ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{
    //初始化label2class缓解了acquire在没有release的情况下就成功的问题
    label2class[0] ="Big_ball";label2class[1] ="Cone"; label2class[2] ="Cone_cylinder";
    label2class[3] ="DT"; label2class[4] ="Small_ball";
    for(auto &item: label2class){
        class2label[item.second] = item.first;
    }
    ui->simulateSignal->setEnabled(false);
    ui->stopListen->setEnabled(false);
    QSemaphore sem(0);
    QMutex lock;
    inferThread =new InferThread(&sem,&sharedQue,&lock);//推理线程
    inferThread->setInferMode("real_time_infer");

    client = new SocketClient();
    connect(client, SIGNAL(sigClassName(int)),this,SLOT(slotShowRealClass(int)));

    connect(inferThread, SIGNAL(sigInferResult(QVector<float>,int,QVariant)),this,SLOT(slotShowInferResult(QVector<float>,int,QVariant)));
    connect(inferThread, SIGNAL(modelAlready()),this,SLOT(slotEnableSimulateSignal()));

    server = new SocketServer(&sem,&sharedQue,&lock,terminal);//监听线程
    // connect(server, SIGNAL(sigSignalVisualize(QVector<float>&)),this,SLOT(slotSignalVisualize(QVector<float>&)));


    connect(ui->startListen, &QPushButton::clicked, this, &MonitorPage::startListen);
    connect(ui->simulateSignal, &QPushButton::clicked, this, &MonitorPage::simulateSend);
    connect(ui->stopListen, &QPushButton::clicked,[this]() { 
        stopSend();
    });
    connect(this, SIGNAL(startOrstop_sig(bool)), client, SLOT(startOrstop_slot(bool)));
}

void MonitorPage::startListen(){
    if(this->choicedDatasetPATH==""){
        QMessageBox::warning(NULL, "实时监测", "监听失败,请先指定数据集");
        //qDebug()<<"modelInfo->selectedType=="<<QString::fromStdString(modelInfo->selectedType);
        return;
    }
    std::string datasetlPath = projectsInfo->pathOfSelectedDataset;
    // 准备CustomDataset，把CustomDataset单个样本的长度传给server
    myDataset = CustomDataset(datasetlPath, false, ".mat", class2label, -1, projectsInfo->modelTypeOfSelectedProject);
    server->setInputLen(myDataset.data[0].size());
    server->start();
    terminal->print("开始监听,等待模型及数据载入中...");
    inferThread->start();
    emit startOrstop_sig(true);
}

void MonitorPage::simulateSend(){
    client->setMyDataset(myDataset);
    client->start();
    ui->stopListen->setEnabled(true);
}

void MonitorPage::stopSend(){
    emit startOrstop_sig(false);
    //停止线程
    client->quit();
    //打断线程中的死循环
//    client->startOrstop_slot();
    client->stopThread();
}


void MonitorPage::refresh(){
    bool ifDataPreProcess=true;
    // 网络输出标签对应类别名称初始化
    std::vector<std::string> comboBoxContents = projectsInfo->classNamesOfSelectedDataset;
    if(comboBoxContents.size()>0){
        for(int i=0;i<comboBoxContents.size();i++){
            label2class[i]=comboBoxContents[i];
            // qDebug()<<"(MonitorPage::refresh) comboBoxContents[i]="<<QString::fromStdString(comboBoxContents[i]);
        }
        for(auto &item: label2class)   class2label[item.second] = item.first;
    }
    // std::map<std::string, int>::iterator iter;
    // iter = class2label.begin();
    // while(iter != class2label.end()) {
    //     std::cout << iter->first << " : " << iter->second << std::endl;
    //     iter++;
    // }
    //如果工程路径变了
    if(
        projectsInfo->getAttri(projectsInfo->dataTypeOfSelectedProject,projectsInfo->nameOfSelectedProject,"Project_Path") != choicedDatasetPATH
    ){
        if(inferThread->isRunning()){//如果已经在跑了 忽视模型更改
            qDebug()<<"(MonitorPage::refresh) inferThread is Running";
            return;
        }
        if(projectsInfo->modelTypeOfSelectedProject=="INCRE") ifDataPreProcess=false;
        choicedModelPATH=projectsInfo->pathOfSelectedModel_forInfer;
        choicedDatasetPATH=projectsInfo->pathOfSelectedDataset;
        inferThread->setClass2LabelMap(class2label);
        //qDebug()<<"(MonitorPage::refresh) class2label.size()=="<<class2label.size();
        inferThread->setParmOfRTI(choicedModelPATH,ifDataPreProcess);//只有小样本是false 既不做预处理
        client->setClass2LabelMap(class2label);
        client->setParmOfRTI(choicedDatasetPATH);//发的数据不做归一化预处理

        ui->datasetname_cil_label->setText(QString::fromStdString(projectsInfo->nameOfSelectedDataset));
        ui->modelname_cil_label->setText(QString::fromStdString(projectsInfo->nameOfSelectedModel_forInfer));
        // qDebug()<<"(MonitorPage::refresh) A  "<<QString::fromStdString(choicedDatasetPATH);
        // qDebug()<<"(MonitorPage::refresh) B  "<<QString::fromStdString(choicedModelPATH);
    }
}

void removeLayout2(QLayout *layout){
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
            removeLayout2(childLayout);
        delete child;
        child = nullptr;
    }
}

void MonitorPage::slotShowInferResult(QVector<float> dataFrameQ, int predIdx, QVariant qv){
    signalVisualize(dataFrameQ);
    Chart *tempChart = new Chart(ui->label_mE_chartGT,"","");//就调用一下它的方法
    //std::vector<float> degrees={0.1,0.1,0.1,0.1,0.2,0.4};
    std::vector<float> degrees=qv.value<std::vector<float>>();
    for(int i=0;i<degrees.size();i++){
        degrees[i]=round(degrees[i] * 100) / 100;
    }
    QString predClass = QString::fromStdString(label2class[predIdx]);
    //terminal->print("Real-time classification results:"+predClass);//连续调用恐怕会有问题
    QWidget *tempWidget=tempChart->drawDisDegreeChart(predClass,degrees,label2class);
    removeLayout2(ui->horizontalLayout_degreeChart2);
    ui->horizontalLayout_degreeChart2->addWidget(tempWidget);
    ui->label_monitor_predClass->setText(QString::fromStdString(label2class[predIdx]));

    if(sigsFromClient.empty()){//如果出现极端情况，连client的类标签都没发过来，就用临时的标签做容错。
        ui->label_monitor_realClass->setText(QString::fromStdString(label2class[fallBackValue]));
        num_fallBackValueUsed++;
    }else{
        for(int i=0;i<num_fallBackValueUsed;i++) sigsFromClient.erase(sigsFromClient.begin());
        num_fallBackValueUsed = 0;
        ui->label_monitor_realClass->setText(QString::fromStdString(label2class[*sigsFromClient.begin()]));
        fallBackValue = *sigsFromClient.begin();
        sigsFromClient.erase(sigsFromClient.begin());
    }

    qDebug()<<"(MonitorPage::slotShowInferResult)"<<ui->label_monitor_realClass->text()<<QString::fromStdString(label2class[predIdx]);
    this->inferedNum ++;
    if(ui->label_monitor_realClass->text() == QString::fromStdString(label2class[predIdx])){
        this->rightNum++;
    }
    qDebug()<<"(MonitorPage::slotShowInferResult) right=="<<this->rightNum<<"   infered_num="<<this->inferedNum;
    QString monitor_acc= QString::number(this->rightNum*100/this->inferedNum);
    qDebug()<<"(MonitorPage::slotShowInferResult) monitor_acc="<<monitor_acc;
    ui->monitor_acc->setText(QString("%1").arg(monitor_acc)+"%");

}

void MonitorPage::slotEnableSimulateSignal(){
    terminal->print("模型已载入可以开始模拟发送");
    ui->simulateSignal->setEnabled(true);
}

void MonitorPage::signalVisualize(QVector<float> dataFrameQ){
    removeLayout2(ui->verticalLayout_hotShow);
    // removeLayout2(ui->verticalLayout_sigShow);

    /*=================单帧==============*/
    QLabel *imageLabel_sig=new QLabel(ui->scrollArea_7);
    std::string currtDataType = projectsInfo->dataTypeOfSelectedProject;
    // qDebug()<<"dataFrameQ.size() === "<<dataFrameQ.size()<<"currtDataType = "<<QString::fromStdString(currtDataType);
    imageLabel_sig->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);  
    Chart *previewChart = new Chart(imageLabel_sig,QString::fromStdString(currtDataType),"");
    previewChart->drawImageWithSingleSignal(imageLabel_sig,dataFrameQ);  

    /*=================热图==============*/
    QLabel *imageLabel_hot=new QLabel(ui->scrollArea_7);
    imageLabel_hot->setBackgroundRole(QPalette::Base);
    imageLabel_hot->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    imageLabel_hot->setScaledContents(true);
    //imageLabel_hot->setStyleSheet("border:2px solid red;");
    QImageReader reader("./colorMap.png");
    reader.setAutoTransform(true);
    const QImage newImage = reader.read();
    if (newImage.isNull()) {
        qDebug()<<"MonitorPage::signalVisualize hotMap not found";
    }
    QImage image;
    image = newImage;
    imageLabel_hot->setPixmap(QPixmap::fromImage(image));

    imageLabel_hot->setMinimumHeight(300);
    imageLabel_sig->setMinimumHeight(300);
    imageLabel_hot->setMaximumWidth(700);
    imageLabel_sig->setMaximumWidth(700);
    ui->verticalLayout_hotShow->addWidget(imageLabel_hot);
    ui->verticalLayout_hotShow->addWidget(imageLabel_sig);

    // ui->verticalLayout_sigShow->addWidget(imageLabel_sig);

}

void MonitorPage::slotShowRealClass(int realLabel){//client触发
    sigsFromClient.push_back(realLabel);
}

MonitorPage::~MonitorPage(){

}
