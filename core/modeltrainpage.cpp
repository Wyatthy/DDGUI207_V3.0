#include "modelTrainPage.h"
#include "qcheckbox.h"
#include "qlistview.h"
#include "qstyleditemdelegate.h"

ModelTrainPage::ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{
    ui->widget_newClass->setVisible(false);
    ui->widget_cilRelate->setVisible(false);
    ui->dataNumPercentEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^0\\.[0-9]{0,1}[1-9]$")));
    ui->preTrainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,3}[1-9]$")));
    ui->trainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));
    ui->trainBatchEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));

    processTrain = new QProcess();
    refreshGlobalInfo();

    connect(processTrain, &QProcess::readyReadStandardOutput, this, &ModelTrainPage::monitorTrainProcess);
    connect(ui->startTrainButton, &QPushButton::clicked, this, &ModelTrainPage::startTrain);
    connect(ui->stopTrainButton,  &QPushButton::clicked, this, &ModelTrainPage::stopTrain);
    connect(ui->editModelButton,  &QPushButton::clicked, this, &ModelTrainPage::editModelFile);
    // connect(ui->trainpage_modelTypeBox, &QComboBox::currentIndexChanged, this, &ModelTrainPage::changeTrainType);
    connect(ui->trainpage_modelTypeBox, SIGNAL(textActivated(QString)), this, SLOT(changeTrainType()));

    connect(ui->pushButton_selectNewData, &QPushButton::clicked, this, &ModelTrainPage::selectNewData);

    connect(ui->pushButton_oldTrain, &QPushButton::clicked, this, &ModelTrainPage::oldClassTrain);

    connect(ui->pushButton_newTrain, &QPushButton::clicked, this, &ModelTrainPage::newClassTrain);


    cliListWidget = new QListWidget;
    cliLineEdit = new QLineEdit;
}




void ModelTrainPage::refreshGlobalInfo(){
    ui->widget_normalTrain->setVisible(true);
    ui->widget_newClass->setVisible(false);
    ui->widget_cilRelate->setVisible(false);
    ui->widget_windowRelate->setVisible(false);
    ui->widget_abfcRelate->setVisible(false);
    dataType = projectsInfo->dataTypeOfSelectedProject;
    modelTypeOfCurrtProject = projectsInfo->modelTypeOfSelectedProject;
    dataDimension = QString::fromStdString(projectsInfo->getAttri(projectsInfo->dataTypeOfSelectedProject, projectsInfo->nameOfSelectedProject, "Dataset_SampleLength")).toInt();
    //Common parm
    ui->trainBatchEdit->setText("16");
    ui->trainEpochEdit->setText("500");
    //below for CIL
    // ui->cil_data_dimension_box->clear();
    // ui->cil_data_dimension_box->addItem(QString::number(128));
    // ui->cil_data_dimension_box->addItem(QString::number(256));
    // ui->cil_data_dimension_box->addItem(QString::number(39));
    ui->dataNumPercentEdit->setText("1.0");
    ui->preTrainEpochEdit->setText("3");
    ui->lineEdit_pretrainBatch->setText("16");
    ui->lineEdit_newclassBatch->setText("16");
    ui->lineEdit_newclassEpoch->setText("3");
    //below for window
    ui->windowsLength->setText("32");
    ui->windowsStep->setText("10");
    //below for abfc
    ui->fea_num->setText("128");
    ui->fea_start->setText("16");
    ui->fea_step->setText("16");

    projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);

    if(projectPath != ""){
        this->choicedDatasetPATH = projectPath+"/train";
        ui->trainPage_dataPathOfSelectedProject->setText(projectPath.split('/').last()+"/train & val");
        // ui->trainPage_modelTypeOfSelectedProject->setText(QString::fromStdString(projectsInfo->modelTypeOfSelectedProject));
    }
    else{
        ui->trainPage_dataPathOfSelectedProject->setText("活动工程未指定");
        // ui->trainPage_modelTypeOfSelectedProject->setText("活动工程未指定");
        this->choicedDatasetPATH = "";
    }
    
    //根据工程的数据类型更新ModelTypeCombobox
    ui->trainpage_modelTypeBox->clear();
    if(projectsInfo->dataTypeOfSelectedProject == "HRRP"){
        ui->trainpage_modelTypeBox->addItem("ATEC");
        ui->trainpage_modelTypeBox->addItem("ABFC");
        ui->trainpage_modelTypeBox->addItem("Baseline_CNN");
        ui->trainpage_modelTypeBox->addItem("Baseline_DNN");
        ui->trainpage_modelTypeBox->addItem("TRAD_Densenet");
        ui->trainpage_modelTypeBox->addItem("TRAD_Resnet50");
        ui->trainpage_modelTypeBox->addItem("TRAD_Resnet101");
        ui->trainpage_modelTypeBox->addItem("TRAD_Mobilenet");
        ui->trainpage_modelTypeBox->addItem("TRAD_Efficientnet");
        ui->trainpage_modelTypeBox->addItem("Incremental_增量学习模型");
    }else if(projectsInfo->dataTypeOfSelectedProject == "RCS" || projectsInfo->dataTypeOfSelectedProject == "IMAGE"){
        ui->trainpage_modelTypeBox->addItem("TRAD_Densenet");
        ui->trainpage_modelTypeBox->addItem("TRAD_Resnet50");
        ui->trainpage_modelTypeBox->addItem("TRAD_Resnet101");
        ui->trainpage_modelTypeBox->addItem("TRAD_Mobilenet");
        ui->trainpage_modelTypeBox->addItem("TRAD_Efficientnet");
    }else if(projectsInfo->dataTypeOfSelectedProject == "FEATURE"){
        ui->trainpage_modelTypeBox->addItem("ABFC");
    }
    refreshTrainResult();
}

void ModelTrainPage::refreshTrainResult(){
    
    //TODO 删除各个tab上的图片

    QString currtTrainAccPic = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/training_accuracy.jpg");
    QString currtValAccPic = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/verification_accuracy.jpg");
    QString currtConfusionPic = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/verification_confusion_matrix.jpg");
    QString currtFeaRelPic = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/features_Accuracy.jpg");
    QString currtFeaWPic = QString::fromStdString(projectsInfo->pathOfSelectedProject + "/features_weights.jpg");
    
    if(this->dirTools->isExist(currtTrainAccPic.toStdString())){
        recvShowPicSignal(QPixmap(currtTrainAccPic), ui->graphicsView_train_trainacc);
    }
    if(this->dirTools->isExist(currtValAccPic.toStdString())){
        recvShowPicSignal(QPixmap(currtValAccPic), ui->graphicsView_train_valacc);
    }
    if(this->dirTools->isExist(currtConfusionPic.toStdString())){
        recvShowPicSignal(QPixmap(currtConfusionPic), ui->graphicsView_train_confusion);
    }
    if(this->dirTools->isExist(currtFeaRelPic.toStdString())){
        recvShowPicSignal(QPixmap(currtFeaRelPic), ui->graphicsView_train_fearel);
    }
    if(this->dirTools->isExist(currtFeaWPic.toStdString())){
        recvShowPicSignal(QPixmap(currtFeaWPic), ui->graphicsView_train_feaw);
    }
    if(modelTypeOfCurrtProject == "ATEC"){
        showATECfeatrend();
    }
}

void ModelTrainPage::changeTrainType(){
    ui->widget_normalTrain->setVisible(true);
    ui->widget_newClass->setVisible(false);
    ui->widget_cilRelate->setVisible(false);
    ui->widget_windowRelate->setVisible(false);
    ui->widget_abfcRelate->setVisible(false);
    shotModelType = ui->trainpage_modelTypeBox->currentText().split("_")[0].toStdString();
    if(ui->trainpage_modelTypeBox->currentText().split("_").size()==2)
        shotModelAlgorithm = ui->trainpage_modelTypeBox->currentText().split("_")[1];    
    // qDebug()<<"get shotModelType"<<QString::fromStdString(shotModelType)<<"     shotModelAlgorithm"<<shotModelAlgorithm;
    QString pathOfTrainDataset = QString::fromStdString(projectsInfo->pathOfSelectedProject) + "/train";
    for(int i=0;i<10;i++){
        ui->tabWidget->removeTab(0);
    }
    if(shotModelType=="ABFC"){
        ui->widget_abfcRelate->setVisible(true);
        ui->tabWidget->addTab(ui->trainpage_fearel,"特征关联性能");
        ui->tabWidget->addTab(ui->trainpage_feaw,"特征权重");
        ui->tabWidget->addTab(ui->trainpage_confusion,"混淆矩阵");
    }else if(shotModelType=="ATEC"){
        ui->tabWidget->addTab(ui->trainpage_trainAcc,"训练集准确率");
        ui->tabWidget->addTab(ui->trainpage_valAcc,"验证集准确率");
        ui->tabWidget->addTab(ui->trainpage_confusion,"混淆矩阵");
        ui->tabWidget->addTab(ui->trainpage_featrend,"特征趋势");

    }else if(shotModelType=="Incremental"){
        ui->widget_normalTrain->setVisible(false);
        ui->widget_newClass->setVisible(true);
        ui->widget_cilRelate->setVisible(true);
        while (cliListWidget->count() > 0){
            QListWidgetItem *item = cliListWidget->takeItem(0);
            delete item;
        }
        QStringList categories = QDir(pathOfTrainDataset).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (int i = 0; i<categories.size(); i++) {
            QListWidgetItem *pItem = new QListWidgetItem(cliListWidget);
            cliListWidget->addItem(pItem);
            pItem->setData(Qt::UserRole, i);
            QCheckBox *pCheckBox = new QCheckBox();
            pCheckBox->setText(categories[i]);
            cliListWidget->addItem(pItem);
            cliListWidget->setItemWidget(pItem, pCheckBox);
        }
        // if (ui->comboBoxasdf->model() != cliListWidget->model()){
        //     ui->comboBoxasdf->setModel(cliListWidget->model());
        //     ui->comboBoxasdf->setView(cliListWidget);
        //     ui->comboBoxasdf->setLineEdit(cliLineEdit);
        //     ui->comboBoxasdf->setMinimumWidth(100);
        //     cliLineEdit->setReadOnly(true);
        // }
        selectedCategories = "";
        for (int i = 0; i < cliListWidget->count()-1; i++) {
            QListWidgetItem *item = cliListWidget->item(i);
            QCheckBox *checkbox = static_cast<QCheckBox *>(cliListWidget->itemWidget(item));
            checkbox->setChecked(true);
            selectedCategories = selectedCategories + checkbox->text(); + ";";
        }
        ui->tabWidget->addTab(ui->trainpage_valAcc,"验证集准确率");
        ui->tabWidget->addTab(ui->trainpage_confusion,"混淆矩阵");
    }
    else {
        ui->tabWidget->addTab(ui->trainpage_trainAcc,"训练集准确率");
        ui->tabWidget->addTab(ui->trainpage_valAcc,"验证集准确率");
        ui->tabWidget->addTab(ui->trainpage_confusion,"混淆矩阵");
    }
    if(dataType == "RCS" || dataType == "IMAGE"){
        ui->widget_windowRelate->setVisible(true);
    }
}


void ModelTrainPage::startTrain(){
    if(shotModelType == ""){
        QMessageBox::information(NULL, "模型训练", "请先从下拉框中选择欲训练模型");
        return;
    }
    this->trainingProjectName = projectsInfo->nameOfSelectedProject;
    this->trainingProjectPath = projectsInfo->pathOfSelectedProject;
    this->trainingDataType = projectsInfo->dataTypeOfSelectedProject;
    qDebug()<<"shotModelType===="<<QString::fromStdString(shotModelType);
    if(shotModelType == "OPTI" || shotModelType == "OPTI_CAM"){
        QMessageBox::information(NULL, "模型训练", "优化模型暂不支持训练");
        return;
    }
    QString datasetPath=this->choicedDatasetPATH;
    QDateTime dateTime(QDateTime::currentDateTime());
    time = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    //Common parm
    batchSize = ui->trainBatchEdit->text();
    epoch = ui->trainEpochEdit->text();
    //below for CIL
    reduce_sample = ui->dataNumPercentEdit->text();
    pretrain_epoch = ui->preTrainEpochEdit->text();
    // cil_data_dimension = ui->cil_data_dimension_box->currentText();
    selectedCategories = "";
    for (int i = 0; i < cliListWidget->count(); i++) {
        QListWidgetItem *item = cliListWidget->item(i);
        //将QWidget 转化为QCheckBox  获取第i个item 的控件
        QCheckBox *checkbox = static_cast<QCheckBox *>(cliListWidget->itemWidget(item));
        if(checkbox->isChecked()){
            QString checkboxStr = checkbox->text();
            selectedCategories = selectedCategories + checkboxStr + ";";
        }
    }
    //below for window
    QString windowsLength = ui->windowsLength->text();
    QString windowsStep = ui->windowsStep->text();
    //below for abfc
    QString fea_num = ui->fea_num->text();
    QString fea_start = ui->fea_start->text();
    QString fea_step = ui->fea_step->text();

    uiInitial();
    //下面根据各种凭据判断当前活动工程使用哪种模型训练
    if(dataType == "RCS"){
        cmd="activate tf24 && python ./api/bashs/RCS/train.py --data_dir "+projectPath+ \
            " --batch_size "+batchSize+" --max_epochs "+epoch+" --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
    }
    else if(dataType == "IMAGE"){
        cmd="activate tf24 && python ./api/bashs/HRRP历程图/train.py --data_dir "+projectPath+ \
            " --batch_size "+batchSize+" --max_epochs "+epoch+" --windows_length "+ windowsLength+" --windows_step "+ windowsStep;
    }
    else if(dataType == "FEATURE"){
        cmd="activate tf24 && python ./api/bashs/ABFC/train.py --data_dir "+projectPath+ \
            " --batch_size "+batchSize+" --max_epochs "+epoch+" --fea_num "+ fea_num+ \
            " --fea_start "+ fea_start + " --data_type FEATURE";
    }
    else if (dataType == "HRRP"){
        if(shotModelType == "TRAD"){
            cmd="activate tf24 && python ./api/bashs/HRRP_Tr/train.py --data_dir "+projectPath+ \
                " --batch_size "+batchSize+" --max_epochs "+epoch;
        }else if(shotModelType == "Baseline"){
            cmd="activate tf24 && python ./api/bashs/baseline/train.py --data_dir "+projectPath+ \
                " --batch_size "+batchSize+" --max_epochs "+epoch;
        }else if(shotModelType == "ATEC"){
            cmd="activate tf24 && python ./api/bashs/ATEC/train.py --data_dir "+projectPath+ \
                " --batch_size "+batchSize+" --max_epochs "+epoch;
        }else if(shotModelType == "ABFC"){
            cmd="activate tf24 && python ./api/bashs/ABFC/train.py --data_dir "+projectPath+ \
                " --batch_size "+batchSize+" --max_epochs "+epoch+" --fea_num "+ fea_num+ \
                " --fea_start "+ fea_start + " --fea_step " + fea_step + " --data_type HRRP";
        }else if(shotModelType == "Incremental"){
            reduce_sample=reduce_sample==""?"1.0":reduce_sample;
            old_class_num=old_class_num==""?"5":old_class_num;
            pretrain_epoch=pretrain_epoch==""?"1":pretrain_epoch;
            epoch=epoch==""?"2":epoch;
            saveModelName=saveModelName==""?"model":saveModelName;

            cmd="activate PT && python ./api/bashs/incremental/main.py --work_dir "+projectPath+ \
            " --time "              + time + \
            " --old_class "         + old_class_num + \
            " --reduce_sample "     + reduce_sample + \
            " --pretrain_epoch "    + pretrain_epoch + \
            " --increment_epoch "   + epoch + \
            " --model_name "        + saveModelName + \
            " --old_class_names " + selectedCategories;
        }
    }
    qDebug()<<"(ModelTrainPage::startTrain) cmd="<<cmd;
    execuCmd(cmd);
    trainningModelType = shotModelType;
}

void ModelTrainPage::uiInitial(){
    ui->startTrainButton->setEnabled(true);
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(0);
    ui->textBrowser->clear();
    // ui->train_img->clear();
    // ui->val_img->clear();
    // ui->confusion_mat->clear();
    // ui->fea_weights->clear();
//    ui->fea_related->clear();
}

void ModelTrainPage::execuCmd(QString cmd){

    if(processTrain->state()==QProcess::Running){
        processTrain->close();
        processTrain->kill();
    }
    showLog=false;
    ui->startTrainButton->setEnabled(false);
    processTrain->setProcessChannelMode(QProcess::MergedChannels);
    processTrain->start("cmd.exe");
    ui->textBrowser->setText("===================Train Starting===================");
    ui->trainProgressBar->setMaximum(0);
    ui->trainProgressBar->setValue(0);
    processTrain->write(cmd.toLocal8Bit() + '\n');
}

void ModelTrainPage::stopTrain(){
    if(currtPID!="")
        QString cmd="taskkill /pid "+ currtPID +"-f";
    else
        QString cmd="\\x03";
    qDebug()<<"currtPID="<<currtPID;

    processTrain->write(cmd.toLocal8Bit() + '\n');
    showLog=false;
    ui->startTrainButton->setEnabled(true);
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(0);
    ui->textBrowser->append("===================Train Stoping===================");
    if(processTrain->state()==QProcess::Running){
        processTrain->close();
        processTrain->kill();
    }
}

void ModelTrainPage::monitorTrainProcess(){
    /* 读取终端输出并显示 */
    QByteArray cmdOut = processTrain->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        QStringList lines = logs.split("\n");
        int len=lines.length();
        for(int i=0;i<len;i++){
            if(lines[i].contains("pid$",Qt::CaseSensitive)){
                currtPID = lines[i].split("pid$")[1];
            }
            if(lines[i].contains("Train Ended",Qt::CaseSensitive)){
                ui->textBrowser->append("===================Train Ended===================");
                showLog=false;
                ui->startTrainButton->setEnabled(true);
                //导入训练好的模型至系统
                
                QString xmlPath = projectPath +"/"+ QString::fromStdString(projectsInfo->nameOfSelectedProject) + ".xml";
                qDebug()<<"xmlPath="<<xmlPath;
                projectsInfo->addProjectFromXML(xmlPath.toStdString());
                projectsInfo->modifyAttri(trainingDataType, trainingProjectName, "Project_Path", trainingProjectPath);
                projectsInfo->writeToXML(projectsInfo->defaultXmlPath);
                // 更新工程信息到xml
                // xml路径
                if (QFile::remove(xmlPath))
                    qDebug()<< "已经移除modelXml";
                else
                    qDebug()<< "modelXml删除失败";
                this->projectsInfo->writePrjInfoToXML(xmlPath.toStdString(),projectsInfo->dataTypeOfSelectedProject,projectsInfo->nameOfSelectedProject);
                showTrianResult();
                if(processTrain->state()==QProcess::Running){
                    processTrain->close();
                    processTrain->kill();
                }
            }
            else if(lines[i].contains(cmd,Qt::CaseSensitive)){
                showLog=true;
            }
            else if(lines[i].contains("Train Failed",Qt::CaseSensitive)){
                ui->startTrainButton->setEnabled(true);
                QDateTime dateTime(QDateTime::currentDateTime());
                ui->textBrowser->append(dateTime.toString("yyyy-MM-dd-hh-mm-ss")+" - 网络模型训练出错：");
                for(i++;i<len;i++){
                    ui->textBrowser->append(lines[i]);
                }
                stopTrain();
            }
            else if(showLog){
                ui->textBrowser->append(lines[i]);
            }
        }
    }
    ui->textBrowser->update();
}


void ModelTrainPage::showTrianResult(){
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    if(trainningModelType=="ABFC"){
        recvShowPicSignal(QPixmap(projectPath+"/training_accuracy.jpg"), ui->graphicsView_train_trainacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_accuracy.jpg"), ui->graphicsView_train_valacc);
        recvShowPicSignal(QPixmap(projectPath+"/features_Accuracy.jpg"), ui->graphicsView_train_fearel);
        recvShowPicSignal(QPixmap(projectPath+"/verification_confusion_matrix.jpg"), ui->graphicsView_train_confusion);
        recvShowPicSignal(QPixmap(projectPath+"/features_weights.jpg"), ui->graphicsView_train_feaw);
    }
    else if(trainningModelType=="Incremental"){
        recvShowPicSignal(QPixmap(projectPath+"/verification_accuracy.jpg"), ui->graphicsView_train_valacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_confusion_matrix.jpg"), ui->graphicsView_train_confusion);
    }
    else if(trainningModelType=="ATEC"){
        recvShowPicSignal(QPixmap(projectPath+"/training_accuracy.jpg"), ui->graphicsView_train_trainacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_accuracy.jpg"), ui->graphicsView_train_valacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_confusion_matrix.jpg"), ui->graphicsView_train_confusion);
        showATECfeatrend();
    }
    else {
        recvShowPicSignal(QPixmap(projectPath+"/training_accuracy.jpg"), ui->graphicsView_train_trainacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_accuracy.jpg"), ui->graphicsView_train_valacc);
        recvShowPicSignal(QPixmap(projectPath+"/verification_confusion_matrix.jpg"), ui->graphicsView_train_confusion);
    }
}

void ModelTrainPage::showATECfeatrend(){
    MatDataProcess_ATECfea *matDataPrcs_atecfea = new MatDataProcess_ATECfea(projectPath.toStdString());
    if(!matDataPrcs_atecfea->ifSucc){
        qDebug()<<"(TrainPage:MatDataProcess_ATECfea) 当前atec工程mapping_feature或者traditional_feature路径还不存在";
        return;
    }
    qDebug()<<"(ModelTrainPage::showATECfeatrend)MatDataProcess_ATECfea正常读到数据";

    int feaNum = matDataPrcs_atecfea->feaNum;
    // for(int i=0;i<3;i++){
    //     qDebug()<<matDataPrcs_atecfea->dataFrames[0][0][i];
    //     qDebug()<<matDataPrcs_atecfea->dataFrames[0][1][i];
    // }
    while (QLayoutItem* item = ui->featureVerticalLayout->takeAt(0)){
        if (QWidget* widget = item->widget())
            widget->deleteLater();
        if (QSpacerItem* spaerItem = item->spacerItem())
            ui->featureVerticalLayout->removeItem(spaerItem);
        delete item;
    }
    for(int i=0;i<feaNum;i++){//画feaNum个图
        QVector<QVector<float>> dataFrame = matDataPrcs_atecfea->dataFrames[i];

        QLabel *imageLabel=new QLabel("Feature"+QString::number(i+1)+":");
        QLabel *imageLabel_sig=new QLabel();
        imageLabel_sig->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        imageLabel_sig->setStyleSheet("border: 3px black");

        Chart *previewChart = new Chart(imageLabel_sig,"","");
        previewChart->diyParams("fea"+QString::number(i),"Sample Index","Value",{"mapping feature","traditional feature"});
        previewChart->drawImageWithMultipleVector(imageLabel_sig,dataFrame,"");
        // previewChart->drawImageWithTwoVector(imageLabel_sig,dataFrame,"fea"+QString::number(i));
        imageLabel_sig->setMinimumHeight(120);
        ui->featureVerticalLayout->addWidget(imageLabel);
        ui->featureVerticalLayout->addWidget(imageLabel_sig);
    }
}
void ModelTrainPage::editModelFile(){
    QString modelFilePath;
    if(dataType == "RCS"){
        modelFilePath="./api/bashs/RCS/train.py";
    }
    else if(dataType == "IMAGE"){
        modelFilePath="./api/bashs/HRRP历程图/train.py";
    }
    else if(dataType == "FEATURE"){
        modelFilePath="./api/bashs/ABFC/train.py";
    }
    else if (dataType == "HRRP"){
        if(shotModelType == "TRAD"){
            modelFilePath="./api/bashs/HRRP_Tr/train.py";
        }else if(shotModelType == "Baseline"){
            modelFilePath="./api/bashs/baseline/train.py";
        }else if(shotModelType == "ATEC"){
            modelFilePath="./api/bashs/ATEC/train.py";
        }else if(shotModelType == "Incremental"){
            modelFilePath="./api/bashs/incremental/train.py";
        }
    }
    QString commd="gvim " + modelFilePath;
    system(commd.toStdString().c_str());
}

void ModelTrainPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    QGraphicsScene *qgraphicsScene = new QGraphicsScene; //要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image);  //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;              //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;           //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}

// 选择新类数据集并且把路径赋值给LineEdit显示
void ModelTrainPage::selectNewData(){
    QString path = QFileDialog::getExistingDirectory(NULL, tr("Open Directory"),
                                                    QDir::currentPath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    // 确定选择了非空路径
    if(path == ""){
        return;
    }
    ui->lineEdit_newDataPath->setText(path);
    this->newClassDatasetPATH = path;
}

void ModelTrainPage::oldClassTrain(){
    this->trainingProjectName = projectsInfo->nameOfSelectedProject;
    this->trainingProjectPath = projectsInfo->pathOfSelectedProject;
    this->trainingDataType = projectsInfo->dataTypeOfSelectedProject;
    qDebug()<<"shotModelType===="<<QString::fromStdString(shotModelType);
    if(shotModelType == "OPTI" || shotModelType == "OPTI_CAM"){
        QMessageBox::information(NULL, "模型训练", "优化模型暂不支持训练");
        return;
    }
    QString datasetPath=this->choicedDatasetPATH;
    QDateTime dateTime(QDateTime::currentDateTime());
    time = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    //Common parm
    batchSize = ui->lineEdit_pretrainBatch->text();
    //below for CIL
    pretrain_epoch = ui->preTrainEpochEdit->text();
    // cil_data_dimension = ui->cil_data_dimension_box->currentText();
    selectedCategories = "";
    for (int i = 0; i < cliListWidget->count(); i++) {
        QListWidgetItem *item = cliListWidget->item(i);
        //将QWidget 转化为QCheckBox  获取第i个item 的控件
        QCheckBox *checkbox = static_cast<QCheckBox *>(cliListWidget->itemWidget(item));
        if(checkbox->isChecked()){
            QString checkboxStr = checkbox->text();
            selectedCategories = selectedCategories + checkboxStr + ";";
        }
    }

    uiInitial();
    //下面根据各种凭据判断当前活动工程使用哪种模型训练
        if(shotModelType == "Incremental"){
            reduce_sample=reduce_sample==""?"1.0":reduce_sample;
            // old_class_num=old_class_num==""?"5":old_class_num;
            pretrain_epoch=pretrain_epoch==""?"1":pretrain_epoch;
            saveModelName=saveModelName==""?"model":saveModelName;
            // QString oldDataPath = projectPath + "/增量学习/旧类数据";
            cmd="activate PT && python ./api/bashs/incremental_403/old_train.py --work_dir "+projectPath+ \
            " --time "              + time + \
            " --batch_size "        + batchSize + \
            " --pretrain_epoch "    + pretrain_epoch + \
            " --model_name "        + saveModelName;
        }
    qDebug()<<"(ModelTrainPage::startTrain) cmd="<<cmd;
    execuCmd(cmd);
    trainningModelType = shotModelType;
}

void ModelTrainPage::newClassTrain(){

    if (newClassDatasetPATH == ""){
        QMessageBox::information(NULL, "模型训练", "请先选择新类数据集");
        return;
    }
    this->trainingProjectName = projectsInfo->nameOfSelectedProject;
    this->trainingProjectPath = projectsInfo->pathOfSelectedProject;
    this->trainingDataType = projectsInfo->dataTypeOfSelectedProject;
    
    qDebug()<<"shotModelType===="<<QString::fromStdString(shotModelType);
    if(shotModelType == "OPTI" || shotModelType == "OPTI_CAM"){
        QMessageBox::information(NULL, "模型训练", "优化模型暂不支持训练");
        return;
    }
    QDateTime dateTime(QDateTime::currentDateTime());
    time = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    //Common parm
    batchSize = ui->lineEdit_newclassBatch->text();
    epoch = ui->lineEdit_newclassEpoch->text();
    //below for CIL
    reduce_sample = ui->dataNumPercentEdit->text();
    // cil_data_dimension = ui->cil_data_dimension_box->currentText();
    selectedCategories = "";
    for (int i = 0; i < cliListWidget->count(); i++) {
        QListWidgetItem *item = cliListWidget->item(i);
        //将QWidget 转化为QCheckBox  获取第i个item 的控件
        QCheckBox *checkbox = static_cast<QCheckBox *>(cliListWidget->itemWidget(item));
        if(checkbox->isChecked()){
            QString checkboxStr = checkbox->text();
            selectedCategories = selectedCategories + checkboxStr + ";";
        }
    }

    uiInitial();
    QString localNewDataPath = projectPath + "/增量学习/新类数据集";
    QString oldAndNewDataPath = projectPath + "/增量学习/新旧类拼接数据集";
    dirTools->copyDir(newClassDatasetPATH, localNewDataPath);
    QString trainPath =  projectPath + "/train";
    // 类别数目是trainPath下的文件夹数量
    QStringList folders = QDir(trainPath).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    int old_class_num = folders.size();
    // 构建新旧类数据集,把旧类数据集下三个文件夹的数据复制到新类数据集下
    QMap<QString, QString> oldClassDataMap;
    oldClassDataMap.insert("train", trainPath);
    oldClassDataMap.insert("val", projectPath + "/val");
    oldClassDataMap.insert("test", projectPath + "/test");

    QMap<QString, QString> newClassDataMap;
    newClassDataMap.insert("train", localNewDataPath + "/train");
    newClassDataMap.insert("val", localNewDataPath + "/val");
    newClassDataMap.insert("test", localNewDataPath + "/test");

    QMap<QString, QString> newOldDataMap;
    newOldDataMap.insert("train", oldAndNewDataPath + "/train");
    newOldDataMap.insert("val", oldAndNewDataPath + "/val");
    newOldDataMap.insert("test", oldAndNewDataPath + "/test");

    // 遍历把旧类和新类三个文件夹train\val\test数据放到oldAndNewDataPath对应文件夹下
    QMapIterator<QString, QString> i(oldClassDataMap);
    dirTools->copyDir(oldClassDataMap.value("train"), newOldDataMap.value("train"));
    dirTools->copyDir(oldClassDataMap.value("val"), newOldDataMap.value("val"));
    dirTools->copyDir(oldClassDataMap.value("test"), newOldDataMap.value("test"));
    dirTools->copyDir(newClassDataMap.value("train"), newOldDataMap.value("train"));
    dirTools->copyDir(newClassDataMap.value("val"), newOldDataMap.value("val"));
    dirTools->copyDir(newClassDataMap.value("test"), newOldDataMap.value("test"));
    

    // localNewDataPath = 
    // 复制新类数据集到本地

    //下面根据各种凭据判断当前活动工程使用哪种模型训练
        if(shotModelType == "Incremental"){
            reduce_sample=reduce_sample==""?"1.0":reduce_sample;
            // old_class_num=old_class_num==""?"5":old_class_num;
            pretrain_epoch=pretrain_epoch==""?"1":pretrain_epoch;
            saveModelName=saveModelName==""?"model":saveModelName;
            cmd="activate PT && python ./api/bashs/incremental_403/new_train.py --work_dir "+localNewDataPath+ \
            " --increment_epoch "  + epoch + \
            " --batch_size "        + batchSize + \
            " --time "              + time + \
            " --reduce_sample "     + reduce_sample + \
            " --old_class "          + QString::number(old_class_num) + \
            " --model_name "        + saveModelName;
        }


    qDebug()<<"(ModelTrainPage::startTrain) cmd="<<cmd;
    execuCmd(cmd);
    trainningModelType = shotModelType;

}