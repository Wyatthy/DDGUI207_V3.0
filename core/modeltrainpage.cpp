#include "modelTrainPage.h"

ModelTrainPage::ModelTrainPage(Ui_MainWindow *main_ui, BashTerminal *bash_terminal, DatasetInfo *globalDatasetInfo,ModelInfo *globalModelInfo, ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    datasetInfo(globalDatasetInfo),
    modelInfo(globalModelInfo),
    projectsInfo(globalProjectInfo)
{

    ui->fewShotWidget->setVisible(false);
    ui->oldClassNumEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,3}$")));
    ui->dataNumPercentEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^0\\.[0-9]{0,1}[1-9]$")));
    ui->preTrainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,3}[1-9]$")));
    ui->trainEpochEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));
    ui->trainBatchEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("^[1-9][0-9]{1,4}[1-9]$")));
    ui->saveModelNameEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9_]+$")));

    ui->oldClassNumEdit->setText("1");
    ui->dataNumPercentEdit->setText("0.9");
    ui->preTrainEpochEdit->setText("100");
    ui->trainEpochEdit->setText("500");
    ui->trainBatchEdit->setText("16");

    ui->saveModelNameEdit->setValidator(new QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9_]+$")));


    processTrain = new QProcess();
    refreshGlobalInfo();

    connect(processTrain, &QProcess::readyReadStandardOutput, this, &ModelTrainPage::monitorTrainProcess);
    connect(ui->startTrainButton, &QPushButton::clicked, this, &ModelTrainPage::startTrain);
    connect(ui->stopTrainButton,  &QPushButton::clicked, this, &ModelTrainPage::stopTrain);
    connect(ui->editModelButton,  &QPushButton::clicked, this, &ModelTrainPage::editModelFile);

}


void ModelTrainPage::refreshGlobalInfo(){
    ui->cil_data_dimension_box->clear();
    ui->cil_data_dimension_box->addItem(QString::number(256));
    ui->cil_data_dimension_box->addItem(QString::number(128));
    ui->cil_data_dimension_box->addItem(QString::number(39));

    QString projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    if(projectPath != ""){
        this->choicedDatasetPATH = projectPath+"/train";
        ui->choosedDataText->setText(projectPath.split('/').last()+"/train");
    }
    else{
        ui->choosedDataText->setText("未指定");
        this->choicedDatasetPATH = "";
    }

    changeTrainType();
}


void ModelTrainPage::changeTrainType(){

    modelType = projectsInfo->modelTypeOfSelectedProject;
    modelName = projectsInfo->modelNameOfSelectedProject;
    ui->tabWidget->removeTab(0);
    ui->tabWidget->removeTab(1);
    ui->tabWidget->removeTab(2);
    if(modelType=="ABFC"){
        ui->fewShotWidget->setVisible(false);
        ui->tabWidget->addTab(ui->tab_2,"特征关联性能");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }
    else if(modelType=="ATEC"){   
        ui->fewShotWidget->setVisible(false);
        ui->tabWidget->addTab(ui->tab,"训练集准确率");
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }
    else if(modelType=="INCRE"){  
        ui->fewShotWidget->setVisible(true);
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }
    else {
        ui->fewShotWidget->setVisible(false);
        ui->tabWidget->addTab(ui->tab,"训练集准确率");
        ui->tabWidget->addTab(ui->tab_2,"验证集准确率");
        ui->tabWidget->addTab(ui->tab_3,"混淆矩阵");
    }
}


void ModelTrainPage::startTrain(){
    if(choicedDatasetPATH==""){
        QMessageBox::warning(NULL,"错误","未选择训练数据集!");
        return;
    }
    QString datasetPath=this->choicedDatasetPATH;
    QDateTime dateTime(QDateTime::currentDateTime());
    time = dateTime.toString("yyyy-MM-dd-hh-mm-ss");
    batchSize = ui->trainBatchEdit->text();
    epoch = ui->trainEpochEdit->text();
    saveModelName = ui->saveModelNameEdit->text();
    //below for CIL
    old_class_num = ui->oldClassNumEdit->text();
    reduce_sample = ui->dataNumPercentEdit->text();
    pretrain_epoch = ui->preTrainEpochEdit->text(); 
    cil_data_dimension = ui->cil_data_dimension_box->currentText();
    
    if(modelType!="INCRE"){
        if(batchSize=="" || epoch=="" || saveModelName==""){
            QMessageBox::warning(NULL,"错误","请检查各项文本框中训练参数是否正确配置!");
            return;
        }
        uiInitial();
        cmd="activate tf24 && python ./api/bashs/hrrp_TRImodel/train.py --data_dir "+choicedDatasetPATH+ \
                        " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName;
        if(modelName == "dnn"){
            cmd = "activate tf24 && python ./api/bashs/baseline/baseline2.py --data_dir "+choicedDatasetPATH+ \
                " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName+" --net DNN";
        }else if(modelName == "cnn"){
            cmd ="activate tf24 && python ./api/bashs/baseline/baseline.py --data_dir "+choicedDatasetPATH+ \
                " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName+" --net CNN";
        }else if(modelName == "densenet121"){
            cmd +=" --net DenseNet121";
        }else if(modelName == "resnet50v2"){
            cmd +=" --net ResNet50V2";
        }else if(modelName == "resnet101"){
            cmd +=" --net ResNet101";
        }else if(modelName == "mobilenet"){
            cmd +=" --net MobileNet";
        }else if(modelName == "efficientnetb0"){
            cmd +=" --net EfficientNetB0";
        }else if(modelName == "abfc"){
            cmd = "activate tf24 && python ./api/bashs/abfc/train.py --data_dir "+choicedDatasetPATH+ \
                " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName;
        }else if(modelName == "atec"){
            cmd = "activate tf24 && python ./api/bashs/atec/main.py --data_dir "+choicedDatasetPATH+ \
                    " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName+ \
                    " --new_data_dir "+"./db/datasets/"+"FEATURE_-"+this->choicedDatasetPATH+"-_36xN_c6";
        }else if(modelName == "rcs"){
            cmd ="activate tf24 && python ./api/bashs/rcs/rcs_densenet.py --data_dir "+choicedDatasetPATH+ \
                    " --time "+time+" --batch_size "+batchSize+" --max_epochs "+epoch+" --model_name "+saveModelName;
        }
    }
    else if(modelType=="INCRE"){     //小样本增量模型训练
        reduce_sample=reduce_sample==""?"1.0":reduce_sample;
        old_class_num=old_class_num==""?"5":old_class_num;
        pretrain_epoch=pretrain_epoch==""?"1":pretrain_epoch;
        epoch=epoch==""?"2":epoch;
        saveModelName=saveModelName==""?"model":saveModelName;

        cmd="activate PT && python ./api/bashs/incremental/main.py --raw_data_path "+choicedDatasetPATH+ \
        " --time "              + time + \
        " --old_class "         + old_class_num + \
        " --reduce_sample "     + reduce_sample + \
        " --pretrain_epoch "    + pretrain_epoch + \
        " --increment_epoch "   + epoch + \
        " --model_name "        + saveModelName + \
        " --data_dimension "    + cil_data_dimension;

    }
    execuCmd(cmd);
}

void ModelTrainPage::uiInitial(){
    ui->startTrainButton->setEnabled(true);
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(0);
    ui->textBrowser->clear();
    ui->train_img->clear();
    ui->val_img->clear();
    ui->confusion_mat->clear();
}

void ModelTrainPage::execuCmd(QString cmd){
  // TODO add code here
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
    QString cmd="\\x03";
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
            QStringList Infos = lines[i].simplified().split(" ");
            if(lines[i].contains("Train Ended",Qt::CaseSensitive)){
                ui->textBrowser->append("===================Train Ended===================");
                showLog=false;
                ui->startTrainButton->setEnabled(true);
                //导入训练好的模型至系统
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
    //TODO
    QDir dir("./db/trainLogs");
    QStringList dirList = dir.entryList(QDir::Dirs);
    foreach (auto dir , dirList){
        if(dir.contains(time)){
            QString wordir    = "./db/trainLogs/"+dir;
            if(modelType=="ABFC"){    
                ui->val_img->setPixmap(QPixmap(wordir+"/features_Accuracy.jpg"));
                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            }
            else if(modelType=="INCRE"){  
                ui->val_img->setPixmap(QPixmap(wordir+"/verification_accuracy.jpg"));
                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            }
            else {
                ui->train_img->setPixmap(QPixmap(wordir+"/training_accuracy.jpg"));
                ui->val_img->setPixmap(QPixmap(wordir+"/verification_accuracy.jpg"));
                ui->confusion_mat->setPixmap(QPixmap(wordir+"/confusion_matrix.jpg"));
            } 
        }
    }
}

void ModelTrainPage::editModelFile(){
    int modelType=ui->modelTypeBox->currentIndex();
    QString modelFilePath;
    switch(modelType){
        case 7:modelFilePath="./api/bashs/abfc/train.py";break;
        case 8:modelFilePath="./api/bashs/atec/net_fea.py";break;
        case 9:modelFilePath="./api/bashs/incremental/model.py";break;
        case 10:modelFilePath="./api/bashs/rcs/rcs_densenet.py";break;
        default:modelFilePath="./api/bashs/hrrp_TRImodel/train.py";
    }
    QString commd="gvim " + modelFilePath;
    system(commd.toStdString().c_str());
}
