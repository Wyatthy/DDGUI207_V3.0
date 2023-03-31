#include "modelCAMPage.h"

#include "./lib/guiLogic/tinyXml/tinyxml.h"
#include "./lib/guiLogic/tools/convertTools.h"
#include "./core/projectsWindow/chart.h"
#include <QGraphicsScene>
#include <QMessageBox>
#include <QFileDialog>
#include <mat.h>
#include <QFile>
#include <QtXml>
#include <QDomDocument>
#include <QDomElement>

using namespace std;


ModelCAMPage::ModelCAMPage(Ui_MainWindow *main_ui,
                             BashTerminal *bash_terminal,
                             ProjectsInfo *globalProjectInfo):
    ui(main_ui),
    terminal(bash_terminal),
    projectsInfo(globalProjectInfo)
{   
    // 设置全局内存限制为1000MB,实现高清图像显示
    QImageReader::setAllocationLimit(1000); 

    // 刷新模型、数据集信息
    refreshGlobalInfo();
    connect(ui->pushButton_CAM_modelConfirm, &QPushButton::clicked, this, &ModelCAMPage::confirmModel);
    
    // 样本选择下拉框信号槽绑定
    connect(ui->comboBox_CAM_stage, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_stage(QString)));
    connect(ui->comboBox_CAM_label, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_label(QString)));
    connect(ui->comboBox_CAM_mat, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_mat(QString)));
    // 各种可视化相关按钮信号槽绑定
    connect(ui->pushButton_CAM_confirmIndex, &QPushButton::clicked, this, &ModelCAMPage::confirmData);
    connect(ui->pushButton_CAM_switchIndex, &QPushButton::clicked, this, &ModelCAMPage::switchIndex);
    connect(ui->pushButton_CAM_nextIndex, &QPushButton::clicked, this, &ModelCAMPage::nextIndex);
    connect(ui->pushButton_CAM_clear, &QPushButton::clicked, this, &ModelCAMPage::clearStructComboBox);
    connect(ui->pushButton_CAM_confirmVis, &QPushButton::clicked, this, &ModelCAMPage::confirmVis);

    // 下拉框信号槽绑定
    connect(ui->comboBox_CAM_L1, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L1(QString)));
    connect(ui->comboBox_CAM_L2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L2(QString)));
    connect(ui->comboBox_CAM_L3, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L3(QString)));
    connect(ui->comboBox_CAM_L4, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L4(QString)));
    connect(ui->comboBox_CAM_L5, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L5(QString)));

    // 多线程的信号槽绑定
    processVis = new QProcess();
    connect(processVis, &QProcess::readyReadStandardOutput, this, &ModelCAMPage::processVisFinished);

    /******************************* 以下同样的实现代码(为了实现两个可视化对比) **********************************/
    // 样本选择下拉框信号槽绑定
    connect(ui->comboBox_CAM_stage_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_stage_2(QString)));
    connect(ui->comboBox_CAM_label_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_label_2(QString)));
    connect(ui->comboBox_CAM_mat_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_mat_2(QString)));
    // 各种可视化相关按钮信号槽绑定
    connect(ui->pushButton_CAM_confirmIndex_2, &QPushButton::clicked, this, &ModelCAMPage::confirmData_2);
    connect(ui->pushButton_CAM_switchIndex_2, &QPushButton::clicked, this, &ModelCAMPage::switchIndex_2);
    connect(ui->pushButton_CAM_nextIndex_2, &QPushButton::clicked, this, &ModelCAMPage::nextIndex_2);
    connect(ui->pushButton_CAM_clear_2, &QPushButton::clicked, this, &ModelCAMPage::clearStructComboBox_2);
    connect(ui->pushButton_CAM_confirmVis_2, &QPushButton::clicked, this, &ModelCAMPage::confirmVis_2);

    // 下拉框信号槽绑定
    connect(ui->comboBox_CAM_L1_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L1_2(QString)));
    connect(ui->comboBox_CAM_L2_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L2_2(QString)));
    connect(ui->comboBox_CAM_L3_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L3_2(QString)));
    connect(ui->comboBox_CAM_L4_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L4_2(QString)));
    connect(ui->comboBox_CAM_L5_2, SIGNAL(textActivated(QString)), this, SLOT(on_comboBox_L5_2(QString)));

    // 多线程的信号槽绑定
    processVis_2 = new QProcess();
    connect(processVis_2, &QProcess::readyReadStandardOutput, this, &ModelCAMPage::processVisFinished_2);
    /********************************************************************************************************/
}

ModelCAMPage::~ModelCAMPage(){

}


void ModelCAMPage::confirmVis(){
    this->choicedCamMethod = ui->comboBox_CAM_camMethod->currentText();
    if(this->choicedMatPATH.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择输入样本!");
        return;
    }
    if(this->choicedModelPATH.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选中模型或不支持该类型模型!");
        return;
    }
    if(this->targetVisLayer.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择可视化隐层!");
        return;
    }
    if(this->choicedCamMethod.isEmpty()){
        QMessageBox::warning(NULL,"错误","未指定可视化CAM方法!");
        return;
    }

    // 激活conda python环境
    if (this->choicedModelSuffix == ".pth"){        // pytorch模型
        this->condaEnvName = "PT";
        this->pythonApiPath = "./api/HRRP_vis_torch/hrrp_cam_torch.py";
    }
    else if(this->choicedModelSuffix == ".hdf5"){   // keras模型
        this->condaEnvName = "tf24";
        this->pythonApiPath = "./api/HRRP_vis_keras/hrrp_cam_keras.py";
    }
    else{
        QMessageBox::warning(NULL,"错误","不支持该类型模型!");
        return;
    }
    std::string dataType = projectsInfo->dataTypeOfSelectedProject;
    QString isRCS = "0";
    if(dataType == "RCS"){
        isRCS = "1";
    }

    // 执行python脚本
    QString activateEnv = "conda activate "+this->condaEnvName+"&&";
    QString command = activateEnv + "python " + this->pythonApiPath+ \
        " --project_path="      +this->projectPath+ \
        " --model_name="        +this->choicedModelName+ \
        " --mat_path="          +this->choicedMatPATH+ \
        " --mat_idx "           +QString::number(this->choicedMatIndexBegin)+ \
                                " "+QString::number(this->choicedMatIndexEnd)+ \
        " --visualize_layer="   +this->targetVisLayer+ \
        " --cam_method="        +this->choicedCamMethod+ \
        " --IMAGE_WINDOWS_LENGTH="+this->windowsLength+ \
        " --IMAGE_WINDOWS_STEP=" +this->windowsStep+ \
        " --RCS="               +isRCS;
    this->camImgsSavePath = this->projectPath+"/CAM_Output/"+ \
                            this->choicedStage+"/"+this->choicedLabel+"/"+ \
                            this->choicedMatName+"/" +\
                            QString::number(this->currMatIndex)+"_"+ \
                            this->choicedCamMethod+".png";
    this->terminal->print(command);
    this->execuCmdProcess(command);
}


// 可视化线程
void ModelCAMPage::execuCmdProcess(QString cmd){
    if(processVis->state()==QProcess::Running){
        processVis->close();
        processVis->kill();
    }
    processVis->setProcessChannelMode(QProcess::MergedChannels);
    processVis->start("cmd.exe");
    ui->progressBar_CAM_visFea->setMaximum(0);
    ui->progressBar_CAM_visFea->setValue(0);
    processVis->write(cmd.toLocal8Bit() + '\n');
}


void ModelCAMPage::processVisFinished(){
    QByteArray cmdOut = processVis->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        terminal->print(logs);
        if(logs.contains("finished")){
            terminal->print("可视化已完成！");
            if(processVis->state()==QProcess::Running){
                processVis->close();
                processVis->kill();
            }
            ui->progressBar_CAM_visFea->setMaximum(100);
            ui->progressBar_CAM_visFea->setValue(100);

            // 加载图像
            ui->label_CAM_camImg->setText(this->choicedCamMethod);
            recvShowPicSignal(QPixmap(this->camImgsSavePath), ui->graphicsView_CAM_camImg);

        }
        if(logs.contains("Error") || logs.contains("Errno")){
            terminal->print("可视化失败！");
            QMessageBox::warning(NULL,"错误","所选隐层不支持决策可视化!");
           ui->progressBar_CAM_visFea->setMaximum(100);
           ui->progressBar_CAM_visFea->setValue(0);
        }
    }
}

void ModelCAMPage::refreshGlobalInfo(){
    // 工程信息更新
    this->projectPath = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    ui->label_CAM_project->setText(QString::fromStdString(projectsInfo->nameOfSelectedProject));
    // 遍历工程下的所有的.hdf5和.pth模型
    vector<string> modelNames;
    dirTools->getFilesplus(modelNames, ".hdf5", this->projectPath.toStdString());
    dirTools->getFilesplus(modelNames, ".pth", this->projectPath.toStdString());
    // 更新模型选择下拉框
    ui->comboBox_CAM_modelChoice->clear();
    ui->comboBox_CAM_modelChoice->addItems(CVS::fromStdVector(modelNames));

    // 更新数据样本下拉框
    ui->comboBox_CAM_stage->clear();
    ui->comboBox_CAM_stage_2->clear();
    vector<string> folderNames = {};
    vector<string> roiNames = {"train", "val", "test"};
    vector<string> stageNames = {};  // folderNames和roiNames取交集
    dirTools->getDirsplus(folderNames, this->projectPath.toStdString());
    set<string> folderSet(folderNames.begin(), folderNames.end());  // 必须先转换为set,否则set_intersection会有问题
    set<string> roiSet(roiNames.begin(), roiNames.end());
    std::set_intersection(folderSet.begin(), folderSet.end(), roiSet.begin(), roiSet.end(), std::back_inserter(stageNames));
    ui->comboBox_CAM_stage->addItems(CVS::fromStdVector(stageNames));
    ui->comboBox_CAM_stage_2->addItems(CVS::fromStdVector(stageNames));

}


void ModelCAMPage::confirmModel(){
    // 获取comboBox_mV_modelChoice的内容
    this->choicedModelName = ui->comboBox_CAM_modelChoice->currentText();
    // 获取模型路径
    this->choicedModelPATH = this->projectPath + "/" + this->choicedModelName;
    // 判断模型是否存在
    if(!dirTools->isExist(this->choicedModelPATH.toStdString())){
        QMessageBox::warning(NULL, "", "模型不存在，请重新选择！");
        return;
    }
    QMessageBox::information(NULL, "", "模型切换为：" + this->choicedModelName);
    // 模型类型，目前可视化仅支持.hdf5和.pth
    this->choicedModelSuffix = "." + this->choicedModelName.split(".").last();
    this->modelStructXmlPath = this->projectPath + "/" + this->choicedModelName.split(".").first() + "_struct.xml";
    this->modelStructImgPath = this->projectPath + "/" + this->choicedModelName.split(".").first() + "_structImage";

    // 判断是否存在模型结构文件*_struct.xml，如果没有则返回
    if (!dirTools->isExist(this->modelStructXmlPath.toStdString())){
        QMessageBox::warning(NULL, "", "模型不支持可视化，请重新选择！");
        return;
    }
    // 更新模型结构下拉框
    qDebug()<<"modelStructXmlPath: "<<this->modelStructXmlPath;
    clearStructComboBox();
    clearStructComboBox_2();
}


void ModelCAMPage::confirmData(){
    // 获取用户输入的索引值
    this->choicedMatIndexBegin = ui->lineEdit_CAM_begin->text().toInt();
    this->choicedMatIndexEnd = ui->lineEdit_CAM_end->text().toInt();
    // 判断数据集是否存在
    if(!this->dirTools->isExist(this->choicedMatPATH.toStdString())){
        QMessageBox::warning(NULL, "数据集问题", "数据集不存在，请重新选择！");
        return;
    }
    // 判断索引值是否合法
    if(this->choicedMatIndexBegin < 0 || this->choicedMatIndexEnd < 0){
        QMessageBox::warning(NULL, "数据索引问题", "索引值不能为负数！");
        return;
    }
    if(this->choicedMatIndexBegin > this->choicedMatIndexEnd){
        QMessageBox::warning(NULL, "数据索引问题", "索引值范围不合法！");
        return;
    }
    // 判断索引值是否超出范围
    if(this->choicedMatIndexBegin > this->maxMatIndex || this->choicedMatIndexEnd > this->maxMatIndex){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出范围！");
        return;
    }

    //绘图
    this->currMatIndex = this->choicedMatIndexBegin;
    ui->lineEdit_CAM_currIndex->setText(QString::number(this->currMatIndex));
    Chart *previewChart = new Chart(ui->label_CAM_choicedImg, QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), this->choicedMatPATH);
    previewChart->drawImage(ui->label_CAM_choicedImg, this->currMatIndex, this->windowsLength.toInt(), this->windowsStep.toInt());

    QMessageBox::information(NULL, "数据切换提醒", "数据切换为：" + this->choicedMatPATH + "\n索引范围为：" + QString::number(this->choicedMatIndexBegin) + " - " + QString::number(this->choicedMatIndexEnd));

}


void ModelCAMPage::switchIndex(){
    // 获取下一个索引值
    this->currMatIndex = ui->lineEdit_CAM_currIndex->text().toInt();
    this->choicedCamMethod = ui->comboBox_CAM_camMethod->currentText();
    // 判断索引值是否合法
    if(this->currMatIndex < this->choicedMatIndexBegin || this->currMatIndex > this->choicedMatIndexEnd){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出所选数据范围！");
        return;
    }
    //绘图
    Chart *previewChart = new Chart(ui->label_CAM_choicedImg, QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), this->choicedMatPATH);
    previewChart->drawImage(ui->label_CAM_choicedImg, this->currMatIndex, this->windowsLength.toInt(), this->windowsStep.toInt());

    // 更新可视化结果
    this->camImgsSavePath = this->projectPath+"/CAM_Output/"+ \
                            this->choicedStage+"/"+this->choicedLabel+"/"+ \
                            this->choicedMatName+"/" +\
                            QString::number(this->currMatIndex)+"_"+ \
                            this->choicedCamMethod+".png";
    if(dirTools->isExist(this->camImgsSavePath.toStdString())){
        recvShowPicSignal(QPixmap(this->camImgsSavePath), ui->graphicsView_CAM_camImg);
    }
}


void ModelCAMPage::nextIndex(){
    this->currMatIndex += 1;
    this->choicedCamMethod = ui->comboBox_CAM_camMethod->currentText();
    // 判断索引值是否合法
    if(this->currMatIndex < this->choicedMatIndexBegin || this->currMatIndex > this->choicedMatIndexEnd){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出所选数据范围！");
        return;
    }
    //绘图
    ui->lineEdit_CAM_currIndex->setText(QString::number(this->currMatIndex));
    Chart *previewChart = new Chart(ui->label_CAM_choicedImg, QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), this->choicedMatPATH);
    previewChart->drawImage(ui->label_CAM_choicedImg, this->currMatIndex, this->windowsLength.toInt(), this->windowsStep.toInt());

    // 更新可视化结果
    this->camImgsSavePath = this->projectPath+"/CAM_Output/"+ \
                            this->choicedStage+"/"+this->choicedLabel+"/"+ \
                            this->choicedMatName+"/" +\
                            QString::number(this->currMatIndex)+"_"+ \
                            this->choicedCamMethod+".png";
    if(dirTools->isExist(this->camImgsSavePath.toStdString())){
        recvShowPicSignal(QPixmap(this->camImgsSavePath), ui->graphicsView_CAM_camImg);
    }
}


void ModelCAMPage::clearStructComboBox(){
    // 判断是否存在模型结构文件*_struct.xml，如果没有则返回
    if (!dirTools->isExist(this->modelStructXmlPath.toStdString())){
        ui->comboBox_CAM_L1->clear();
        ui->comboBox_CAM_L2->clear();
        ui->comboBox_CAM_L3->clear();
        ui->comboBox_CAM_L4->clear();
        ui->comboBox_CAM_L5->clear();

        return;
    } 
    // 初始化第一个下拉框
    QStringList L1Layers;
    loadModelStruct_L1(L1Layers, this->choicedLayer_2);
    ui->comboBox_CAM_L1->clear();
    ui->comboBox_CAM_L1->addItems(L1Layers);
    ui->comboBox_CAM_L2->clear();
    ui->comboBox_CAM_L3->clear();
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L5->clear();

    this->choicedLayer["L1"] = "NULL";
    this->choicedLayer["L2"] = "NULL";
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    refreshVisInfo();
}


void ModelCAMPage::refreshVisInfo(){
    // 提取目标层信息的特定格式
    QString targetVisLayer = "";
    if(this->choicedModelSuffix==".pth"){
        vector<string> tmpList = {"L2", "L3"};
        for(auto &layer : tmpList){
            if(this->choicedLayer[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer[layer]);
            }
            else{
                if(this->choicedLayer[layer][0] == '_'){
                    targetVisLayer += QString::fromStdString("["+this->choicedLayer[layer].substr(1)+"]");
                }
                else{
                    targetVisLayer += QString::fromStdString("."+this->choicedLayer[layer]);
                }
            }
        }
        this->targetVisLayer = targetVisLayer.replace("._", ".");
    }
    else if(this->choicedModelSuffix==".hdf5"){
        vector<string> tmpList = {"L2", "L3", "L4", "L5"};
        for(auto &layer : tmpList){
            if(this->choicedLayer[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer[layer]);
            }
            else{
                targetVisLayer += QString::fromStdString("_"+this->choicedLayer[layer]);
            }
        }
        this->targetVisLayer = targetVisLayer.replace("__", "_");
    }

    ui->label_CAM_visLayer->setText(this->targetVisLayer);

    // 加载相应的预览图像
    QString imgPath = this->modelStructImgPath + "/";
    if(this->targetVisLayer == ""){
        imgPath += "framework.png";
    }
    else{
        imgPath = imgPath + this->targetVisLayer + ".png"; 
    }
    if(this->dirTools->isExist(imgPath.toStdString())){
        QImageReader::setAllocationLimit(1000); // 设置全局内存限制为1000MB
        QPixmap img(imgPath);
        if(this->choicedModelSuffix == ".hdf5"){
            recvShowPicSignal(img.transformed(QTransform().rotate(-90)), ui->graphicsView_CAM_modelImg);
        }
        else if(this->choicedModelSuffix == ".pth"){
            recvShowPicSignal(img, ui->graphicsView_CAM_modelImg);
        }
    }
}


void ModelCAMPage::on_comboBox_L1(QString choicedLayer){
    this->choicedLayer["L1"] = choicedLayer.toStdString();
    this->choicedLayer["L2"] = "NULL";
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L2(nextLayers, this->choicedLayer);
    ui->comboBox_CAM_L2->clear();
    ui->comboBox_CAM_L2->addItems(nextLayers);
    ui->comboBox_CAM_L3->clear();
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L2(QString choicedLayer){
    this->choicedLayer["L2"] = choicedLayer.toStdString();
    this->choicedLayer["L3"] = "NULL";
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L3(nextLayers, this->choicedLayer);
    ui->comboBox_CAM_L3->clear();
    ui->comboBox_CAM_L3->addItems(nextLayers);
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L3(QString choicedLayer){
    this->choicedLayer["L3"] = choicedLayer.toStdString();
    this->choicedLayer["L4"] = "NULL";
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L4(nextLayers, this->choicedLayer);
    ui->comboBox_CAM_L4->clear();
    ui->comboBox_CAM_L4->addItems(nextLayers);
    ui->comboBox_CAM_L5->clear();
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L4(QString choicedLayer){
    this->choicedLayer["L4"] = choicedLayer.toStdString();
    this->choicedLayer["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L5(nextLayers, this->choicedLayer);
    ui->comboBox_CAM_L5->clear();
    ui->comboBox_CAM_L5->addItems(nextLayers);
    refreshVisInfo();
}

void ModelCAMPage::on_comboBox_L5(QString choicedLayer){
    this->choicedLayer["L5"] = choicedLayer.toStdString();
    refreshVisInfo();
}


void ModelCAMPage::on_comboBox_stage(QString choicedStage){
    this->choicedStage = choicedStage;
    // 扫描工程目录相应stage下的文件夹，将文件夹作为label下拉框的内容
    vector<string> labelNames = {};
    dirTools->getDirsplus(labelNames, (this->projectPath+"/"+this->choicedStage).toStdString());
    ui->comboBox_CAM_label->clear();
    ui->comboBox_CAM_label->addItems(CVS::fromStdVector(labelNames));
}

void ModelCAMPage::on_comboBox_label(QString choicedLabel){
    this->choicedLabel = choicedLabel;
    // 扫描相应类别label目录下的.mat文件，将其作为mat下拉框的内容
    vector<string> matNames = {};
    dirTools->getFilesplus(matNames, ".mat", (this->projectPath+"/"+this->choicedStage+"/"+this->choicedLabel).toStdString());
    ui->comboBox_CAM_mat->clear();
    ui->comboBox_CAM_mat->addItems(CVS::fromStdVector(matNames));
}

void ModelCAMPage::on_comboBox_mat(QString choicedMat){
    this->choicedMatName = choicedMat;
    this->choicedMatPATH = this->projectPath+"/"+this->choicedStage+"/"+this->choicedLabel+"/"+choicedMat;
    if(dirTools->isExist(this->choicedMatPATH.toStdString())){
        MATFile* pMatFile = NULL;
        mxArray* pMxArray = NULL;
        pMatFile = matOpen(this->choicedMatPATH.toStdString().c_str(), "r");
        if(!pMatFile){qDebug()<<"(ModelEvnameOfMatFilealPage::takeSample)文件指针空!";return;}
        pMxArray = matGetNextVariable(pMatFile, NULL);
        if(!pMxArray){
            qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!";
            return;
        }
        int N = mxGetN(pMxArray);  //N 列数
        // 给用户提示样本范围
        ui->lineEdit_CAM_begin->clear();
        ui->lineEdit_CAM_end->clear();
        ui->lineEdit_CAM_begin->setText("1");
        // 加历程图相关功能
        std::string dataType = projectsInfo->dataTypeOfSelectedProject;
        if(dataType == "IMAGE" || dataType == "RCS"){
            this->windowsLength = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsLength"]);
            this->windowsStep = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsStep"]);
            if(this->windowsLength == "0" || this->windowsStep == "0" || this->windowsLength == "" || this->windowsStep == ""){
                // 模型未训练，不能可视化
                QMessageBox::warning(NULL, "警告", "模型未训练，不能可视化！");
                return;
            }
            int sampleNum = (N - this->windowsLength.toInt())/this->windowsStep.toInt() + 1;
            ui->lineEdit_CAM_end->setText(QString::number(sampleNum));
            this->maxMatIndex = sampleNum;
        }
        else{
            this->windowsLength = "0";
            this->windowsStep = "0";
            ui->lineEdit_CAM_end->setText(QString::number(N));
            this->maxMatIndex = N;
        }
    }
}


void ModelCAMPage::loadModelStruct_L1(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers){
    QFile file(this->modelStructXmlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        qDebug() << "Could not load the modelStruct .xml file. Error: Failed to open file";
        exit(1);
    }
    QDomDocument datasetInfoDoc;
    datasetInfoDoc.setContent(&file);
    file.close();

    QDomElement RootElement = datasetInfoDoc.documentElement();    //根元素, Info
    //遍历一级根结点
    for(QDomElement currL1Ele = RootElement.firstChildElement(); !currL1Ele.isNull(); currL1Ele = currL1Ele.nextSiblingElement()){
        // cout<<"----->"<<currL1Ele->tagName().toStdString()<<endl;
        currLayers.append(currL1Ele.tagName());
    }
}



void ModelCAMPage::loadModelStruct_L2(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers){
    QFile datasetInfoDoc(this->modelStructXmlPath);
    if(!datasetInfoDoc.open(QIODevice::ReadOnly | QIODevice::Text)){
        qDebug() << "Could not open the modelStruct .xml file.";
        return;
    }

    QDomDocument doc;
    if (!doc.setContent(&datasetInfoDoc)) {
        datasetInfoDoc.close();
        qDebug() << "Could not parse the modelStruct .xml file.";
        return;
    }
    datasetInfoDoc.close();

    QDomElement RootElement = doc.documentElement();	//根元素, Info
    //遍历一级根结点
    for(QDomElement currL1Ele = RootElement.firstChildElement(); !currL1Ele.isNull(); currL1Ele = currL1Ele.nextSiblingElement()){
        if(currL1Ele.tagName().toStdString() == choicedLayers["L1"]){
            // 遍历二级子节点
            for(QDomElement currL2Ele=currL1Ele.firstChildElement(); !currL2Ele.isNull(); currL2Ele=currL2Ele.nextSiblingElement()){
                // cout<<"---->"<<currL2Ele->Value()<<endl;
                currLayers.append(currL2Ele.tagName());
            }
        }

    }
}


void ModelCAMPage::loadModelStruct_L3(QStringList& currLayers, std::map<std::string, std::string> &choicedLayers) {
    QDomDocument datasetInfoDoc;
    QFile file(this->modelStructXmlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Could not load the modelStruct .xml file. Error:" << file.errorString();
        exit(1);
    }

    if (!datasetInfoDoc.setContent(&file)) {
        file.close();
        qDebug() << "Could not parse the modelStruct .xml file.";
        exit(1);
    }
    file.close();

    QDomElement rootElement = datasetInfoDoc.documentElement();  //根元素 Info
    //遍历一级根结点
    for (QDomElement currL1Ele = rootElement.firstChildElement(); !currL1Ele.isNull(); currL1Ele = currL1Ele.nextSiblingElement()) {
        if (currL1Ele.tagName().toStdString() == choicedLayers["L1"]) {
            // 遍历二级子节点
            for (QDomElement currL2Ele = currL1Ele.firstChildElement(); !currL2Ele.isNull(); currL2Ele = currL2Ele.nextSiblingElement()) {
                if (currL2Ele.tagName().toStdString() == choicedLayers["L2"]) {
                    // 遍历三级子节点
                    for (QDomElement currL3Ele = currL2Ele.firstChildElement(); !currL3Ele.isNull(); currL3Ele = currL3Ele.nextSiblingElement()) {
                        // cout<<"---->"<<currL3Ele->tagName()<<endl;
                        currLayers.append(currL3Ele.tagName());
                    }
                }
            }
        }
    }
}


void ModelCAMPage::loadModelStruct_L4(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers) {
    QFile file(this->modelStructXmlPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Could not open the modelStruct .xml file.";
        exit(1);
    }

    QDomDocument doc;
    if (!doc.setContent(&file)) {
        qDebug() << "Could not parse the modelStruct .xml file.";
        exit(1);
    }

    QDomElement root = doc.documentElement();
    //遍历一级根结点
    for (QDomElement currL1Ele = root.firstChildElement(); !currL1Ele.isNull(); currL1Ele = currL1Ele.nextSiblingElement()) {
        if (currL1Ele.tagName().toStdString() == choicedLayers["L1"]) {
            // 遍历二级子节点
            for (QDomElement currL2Ele = currL1Ele.firstChildElement(); !currL2Ele.isNull(); currL2Ele = currL2Ele.nextSiblingElement()) {
                if (currL2Ele.tagName().toStdString() == choicedLayers["L2"]) {
                    // 遍历三级子节点
                    for (QDomElement currL3Ele = currL2Ele.firstChildElement(); !currL3Ele.isNull(); currL3Ele = currL3Ele.nextSiblingElement()) {
                        if (currL3Ele.tagName().toStdString() == choicedLayers["L3"]) {
                            // 遍历四级子节点
                            for (QDomElement currL4Ele = currL3Ele.firstChildElement(); !currL4Ele.isNull(); currL4Ele = currL4Ele.nextSiblingElement()) {
                                // cout<<"---->"<<currL4Ele->tagName()<<endl;
                                currLayers.append(currL4Ele.tagName());
                            }
                        }
                    }
                }
            }
        }
    }
    file.close();
}


void ModelCAMPage::loadModelStruct_L5(QStringList &currLayers, std::map<std::string, std::string> &choicedLayers){
    QFile xmlFile(this->modelStructXmlPath);
    if (!xmlFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Could not open the modelStruct .xml file:" << xmlFile.errorString();
        exit(1);
    }

    QDomDocument datasetInfoDoc;
    if (!datasetInfoDoc.setContent(&xmlFile)) {
        qDebug() << "Could not parse the modelStruct .xml file.";
        exit(1);
    }

    QDomElement rootElement = datasetInfoDoc.documentElement();	//根元素, Info
    //遍历一级根结点
    for(QDomNode currL1Node = rootElement.firstChild(); !currL1Node.isNull(); currL1Node = currL1Node.nextSibling()){
        QDomElement currL1Ele = currL1Node.toElement();
        if(currL1Ele.tagName().toStdString() == choicedLayers["L1"]){
            // 遍历二级子节点
            for(QDomNode currL2Node=currL1Ele.firstChild(); !currL2Node.isNull(); currL2Node=currL2Node.nextSibling()){
                QDomElement currL2Ele = currL2Node.toElement();
                if(currL2Ele.tagName().toStdString() == choicedLayers["L2"]){
                    // 遍历三级子节点
                    for(QDomNode currL3Node=currL2Ele.firstChild(); !currL3Node.isNull(); currL3Node=currL3Node.nextSibling()){
                        QDomElement currL3Ele = currL3Node.toElement();
                        if(currL3Ele.tagName().toStdString() == choicedLayers["L3"]){
                            // 遍历四级子节点
                            for(QDomNode currL4Node=currL3Ele.firstChild(); !currL4Node.isNull(); currL4Node=currL4Node.nextSibling()){
                                QDomElement currL4Ele = currL4Node.toElement();
                                if(currL4Ele.tagName().toStdString() == choicedLayers["L4"]){
                                    // 遍历五级子节点
                                    for(QDomNode currL5Node=currL4Ele.firstChild(); !currL5Node.isNull(); currL5Node=currL5Node.nextSibling()){
                                        QDomElement currL5Ele = currL5Node.toElement();
                                        currLayers.append(currL5Ele.tagName());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void ModelCAMPage::recvShowPicSignal(QPixmap image, QGraphicsView *graphicsView){
    QGraphicsScene *qgraphicsScene = new QGraphicsScene;//要用QGraphicsView就必须要有QGraphicsScene搭配着用
    all_Images[graphicsView] = new ImageWidget(&image); //实例化类ImageWidget的对象m_Image，该类继承自QGraphicsItem，是自定义类
    int nwith = graphicsView->width()*0.95;                  //获取界面控件Graphics View的宽度
    int nheight = graphicsView->height()*0.95;               //获取界面控件Graphics View的高度
    all_Images[graphicsView]->setQGraphicsViewWH(nwith, nheight);//将界面控件Graphics View的width和height传进类m_Image中
    qgraphicsScene->addItem(all_Images[graphicsView]);           //将QGraphicsItem类对象放进QGraphicsScene中
    graphicsView->setSceneRect(QRectF(-(nwith/2), -(nheight/2),nwith,nheight));//使视窗的大小固定在原始大小，不会随图片的放大而放大（默认状态下图片放大的时候视窗两边会自动出现滚动条，并且视窗内的视野会变大），防止图片放大后重新缩小的时候视窗太大而不方便观察图片
    graphicsView->setScene(qgraphicsScene); //Sets the current scene to scene. If scene is already being viewed, this function does nothing.
    graphicsView->setFocus();               //将界面的焦点设置到当前Graphics View控件
}



/**************** 以下同样的实现代码(为了实现两个可视化对比) ****************/
void ModelCAMPage::confirmData_2(){
    // 获取用户输入的索引值
    this->choicedMatIndexBegin_2 = ui->lineEdit_CAM_begin_2->text().toInt();
    this->choicedMatIndexEnd_2 = ui->lineEdit_CAM_end_2->text().toInt();
    // 判断数据集是否存在
    if(!this->dirTools->isExist(this->choicedMatPATH_2.toStdString())){
        QMessageBox::warning(NULL, "数据集问题", "数据集不存在，请重新选择！");
        return;
    }
    // 判断索引值是否合法
    if(this->choicedMatIndexBegin_2 < 0 || this->choicedMatIndexEnd_2 < 0){
        QMessageBox::warning(NULL, "数据索引问题", "索引值不能为负数！");
        return;
    }
    if(this->choicedMatIndexBegin_2 > this->choicedMatIndexEnd_2){
        QMessageBox::warning(NULL, "数据索引问题", "索引值范围不合法！");
        return;
    }
    // 判断索引值是否超出范围
    if(this->choicedMatIndexBegin_2 > this->maxMatIndex_2 || this->choicedMatIndexEnd_2 > this->maxMatIndex_2){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出范围！");
        return;
    }

    //绘图
    this->currMatIndex_2 = this->choicedMatIndexBegin_2;
    ui->lineEdit_CAM_currIndex_2->setText(QString::number(this->currMatIndex_2));
    Chart *previewChart = new Chart(
        ui->label_CAM_choicedImg_2, 
        QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), 
        this->choicedMatPATH_2
    );
    previewChart->drawImage(ui->label_CAM_choicedImg_2, this->currMatIndex_2, this->windowsLength.toInt(), this->windowsStep.toInt());

    QMessageBox::information(NULL, "数据切换提醒", "数据切换为：" + this->choicedMatPATH_2 + \
            "\n索引范围为：" + QString::number(this->choicedMatIndexBegin_2) + " - " + \
            QString::number(this->choicedMatIndexEnd_2));

}

void ModelCAMPage::switchIndex_2(){
    // 获取用户输入的索引值
    this->choicedCamMethod_2 = ui->comboBox_CAM_camMethod_2->currentText();
    this->currMatIndex_2 = ui->lineEdit_CAM_currIndex_2->text().toInt();
    // 判断索引值是否合法
    if(this->currMatIndex_2 < this->choicedMatIndexBegin_2 || this->currMatIndex_2 > this->choicedMatIndexEnd_2){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出所选数据范围！");
        return;
    }
    //绘图
    Chart *previewChart = new Chart(ui->label_CAM_choicedImg_2, \
        QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), this->choicedMatPATH_2);
    previewChart->drawImage(ui->label_CAM_choicedImg_2, this->currMatIndex_2, this->windowsLength.toInt(), this->windowsStep.toInt());

    // 更新可视化结果
    this->camImgsSavePath_2 = this->projectPath+"/CAM_Output/"+ \
                              this->choicedStage_2+"/"+this->choicedLabel_2+"/"+ \
                              this->choicedMatName_2+"/" +\
                              QString::number(this->currMatIndex_2)+"_"+ \
                              this->choicedCamMethod_2+".png";
    if(dirTools->isExist(this->camImgsSavePath_2.toStdString())){
        recvShowPicSignal(QPixmap(this->camImgsSavePath_2), ui->graphicsView_CAM_camImg_2);
    }
}

void ModelCAMPage::nextIndex_2(){
    // 获取下一个索引值
    this->currMatIndex_2 += 1;
    this->choicedCamMethod_2 = ui->comboBox_CAM_camMethod_2->currentText();
    // 判断索引值是否合法
    if(this->currMatIndex_2 < this->choicedMatIndexBegin_2 || this->currMatIndex_2 > this->choicedMatIndexEnd_2){
        QMessageBox::warning(NULL, "数据索引问题", "索引值超出所选数据范围！");
        return;
    }
    //绘图
    ui->lineEdit_CAM_currIndex_2->setText(QString::number(this->currMatIndex_2));
    Chart *previewChart = new Chart(ui->label_CAM_choicedImg_2, \
        QString::fromStdString(projectsInfo->dataTypeOfSelectedProject), this->choicedMatPATH_2);
    previewChart->drawImage(ui->label_CAM_choicedImg_2, this->currMatIndex_2, this->windowsLength.toInt(), this->windowsStep.toInt());

    // 更新可视化结果
    this->camImgsSavePath_2 = this->projectPath+"/CAM_Output/"+ \
                              this->choicedStage_2+"/"+this->choicedLabel_2+"/"+ \
                              this->choicedMatName_2+"/" +\
                              QString::number(this->currMatIndex_2)+"_"+ \
                              this->choicedCamMethod_2+".png";
    if(dirTools->isExist(this->camImgsSavePath_2.toStdString())){
        recvShowPicSignal(QPixmap(this->camImgsSavePath_2), ui->graphicsView_CAM_camImg_2);
    }
}

void ModelCAMPage::refreshVisInfo_2(){
    // 提取目标层信息的特定格式
    QString targetVisLayer = "";
    if(this->choicedModelSuffix==".pth"){
        vector<string> tmpList = {"L2", "L3"};
        for(auto &layer : tmpList){
            if(this->choicedLayer_2[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer_2[layer]);
            }
            else{
                if(this->choicedLayer_2[layer][0] == '_'){
                    targetVisLayer += QString::fromStdString("["+this->choicedLayer_2[layer].substr(1)+"]");
                }
                else{
                    targetVisLayer += QString::fromStdString("."+this->choicedLayer_2[layer]);
                }
            }
        }
        this->targetVisLayer_2 = targetVisLayer.replace("._", ".");
    }
    else if(this->choicedModelSuffix==".hdf5"){
        vector<string> tmpList = {"L2", "L3", "L4", "L5"};
        for(auto &layer : tmpList){
            if(this->choicedLayer_2[layer] == "NULL"){
                continue;
            }
            if(layer == "L2"){
                targetVisLayer += QString::fromStdString(this->choicedLayer_2[layer]);
            }
            else{
                targetVisLayer += QString::fromStdString("_"+this->choicedLayer_2[layer]);
            }
        }
        this->targetVisLayer_2 = targetVisLayer.replace("__", "_");
    }

    ui->label_CAM_visLayer_2->setText(this->targetVisLayer_2);

    // 加载相应的预览图像
    QString imgPath = this->modelStructImgPath + "/";
    if(this->targetVisLayer_2 == ""){
        imgPath += "framework.png";
    }
    else{
        imgPath = imgPath + this->targetVisLayer_2 + ".png"; 
    }
    if(this->dirTools->isExist(imgPath.toStdString())){
        QImageReader::setAllocationLimit(1000); // 设置全局内存限制为1000MB
        QPixmap img(imgPath);
        if(this->choicedModelSuffix == ".hdf5"){
            recvShowPicSignal(img.transformed(QTransform().rotate(-90)), ui->graphicsView_CAM_modelImg);
        }
        else if(this->choicedModelSuffix == ".pth"){
            recvShowPicSignal(img, ui->graphicsView_CAM_modelImg);
        }
    }
}

void ModelCAMPage::clearStructComboBox_2(){
    // 判断是否存在模型结构文件*_struct.xml，如果没有则返回
    if (!dirTools->isExist(this->modelStructXmlPath.toStdString())){
        ui->comboBox_CAM_L1_2->clear();
        ui->comboBox_CAM_L2_2->clear();
        ui->comboBox_CAM_L3_2->clear();
        ui->comboBox_CAM_L4_2->clear();
        ui->comboBox_CAM_L5_2->clear();

        return;
    } 
    // 初始化第一个下拉框
    QStringList L1Layers;
    loadModelStruct_L1(L1Layers, this->choicedLayer);
    ui->comboBox_CAM_L1_2->clear();
    ui->comboBox_CAM_L1_2->addItems(L1Layers);
    ui->comboBox_CAM_L2_2->clear();
    ui->comboBox_CAM_L3_2->clear();
    ui->comboBox_CAM_L4_2->clear();
    ui->comboBox_CAM_L5_2->clear();

    this->choicedLayer_2["L1"] = "NULL";
    this->choicedLayer_2["L2"] = "NULL";
    this->choicedLayer_2["L3"] = "NULL";
    this->choicedLayer_2["L4"] = "NULL";
    this->choicedLayer_2["L5"] = "NULL";

    refreshVisInfo_2();
}


void ModelCAMPage::confirmVis_2(){
    this->choicedCamMethod_2 = ui->comboBox_CAM_camMethod_2->currentText();
    if(this->choicedMatPATH_2.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择输入样本!");
        return;
    }
    if(this->choicedModelPATH.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选中模型或不支持该类型模型!");
        return;
    }
    if(this->targetVisLayer_2.isEmpty()){
        QMessageBox::warning(NULL,"错误","未选择可视化隐层!");
        return;
    }
    if(this->choicedCamMethod_2.isEmpty()){
        QMessageBox::warning(NULL,"错误","未指定可视化CAM方法!");
        return;
    }

    // 激活conda python环境
    if (this->choicedModelSuffix == ".pth"){        // pytorch模型
        this->condaEnvName = "PT";
        this->pythonApiPath = "./api/HRRP_vis_torch/hrrp_cam_torch.py";
    }
    else if(this->choicedModelSuffix == ".hdf5"){   // keras模型
        this->condaEnvName = "tf24";
        this->pythonApiPath = "./api/HRRP_vis_keras/hrrp_cam_keras.py";
    }
    else{
        QMessageBox::warning(NULL,"错误","不支持该类型模型!");
        return;
    }
    std::string dataType = projectsInfo->dataTypeOfSelectedProject;
    QString isRCS = "0";
    if(dataType == "RCS"){
        isRCS = "1";
    }

    // 执行python脚本
    QString activateEnv = "conda activate "+this->condaEnvName+"&&";
    QString command = activateEnv + "python " + this->pythonApiPath+ \
        " --project_path="      +this->projectPath+ \
        " --model_name="        +this->choicedModelName+ \
        " --mat_path="          +this->choicedMatPATH_2+ \
        " --mat_idx "           +QString::number(this->choicedMatIndexBegin_2)+ \
                                " "+QString::number(this->choicedMatIndexEnd_2)+ \
        " --visualize_layer="   +this->targetVisLayer_2+ \
        " --cam_method="        +this->choicedCamMethod_2+ \
        " --IMAGE_WINDOWS_LENGTH="+this->windowsLength+ \
        " --IMAGE_WINDOWS_STEP=" +this->windowsStep+ \
        " --RCS="               +isRCS;
    this->camImgsSavePath_2 = this->projectPath+"/CAM_Output/"+ \
                            this->choicedStage_2+"/"+this->choicedLabel_2+"/"+ \
                            this->choicedMatName_2+"/" +\
                            QString::number(this->currMatIndex_2)+"_"+ \
                            this->choicedCamMethod_2+".png";
    this->terminal->print(command);
    this->execuCmdProcess_2(command);
}

void ModelCAMPage::execuCmdProcess_2(QString cmd){
    if(processVis_2->state()==QProcess::Running){
        processVis_2->close();
        processVis_2->kill();
    }
    processVis_2->setProcessChannelMode(QProcess::MergedChannels);
    processVis_2->start("cmd.exe");
    ui->progressBar_CAM_visFea_2->setMaximum(0);
    ui->progressBar_CAM_visFea_2->setValue(0);
    processVis_2->write(cmd.toLocal8Bit() + '\n');
}

void ModelCAMPage::processVisFinished_2(){
    QByteArray cmdOut = processVis_2->readAllStandardOutput();
    if(!cmdOut.isEmpty()){
        QString logs=QString::fromLocal8Bit(cmdOut);
        terminal->print(logs);
        if(logs.contains("finished")){
            terminal->print("可视化已完成！");
            if(processVis_2->state()==QProcess::Running){
                processVis_2->close();
                processVis_2->kill();
            }
            ui->progressBar_CAM_visFea_2->setMaximum(100);
            ui->progressBar_CAM_visFea_2->setValue(100);

            // 加载图像
            ui->label_CAM_camImg_2->setText(this->choicedCamMethod_2);
            recvShowPicSignal(QPixmap(this->camImgsSavePath_2), ui->graphicsView_CAM_camImg_2);

        }
        if(logs.contains("Error") || logs.contains("Errno")){
            terminal->print("可视化失败！");
            QMessageBox::warning(NULL,"错误","所选隐层不支持决策可视化!");
           ui->progressBar_CAM_visFea_2->setMaximum(100);
           ui->progressBar_CAM_visFea_2->setValue(0);
        }
    }
}



// 5级下拉框相关槽接口，过于暴力，不优雅 // TODO
void ModelCAMPage::on_comboBox_L1_2(QString choicedLayer){
    this->choicedLayer_2["L1"] = choicedLayer.toStdString();
    this->choicedLayer_2["L2"] = "NULL";
    this->choicedLayer_2["L3"] = "NULL";
    this->choicedLayer_2["L4"] = "NULL";
    this->choicedLayer_2["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L2(nextLayers, this->choicedLayer_2);
    ui->comboBox_CAM_L2_2->clear();
    ui->comboBox_CAM_L2_2->addItems(nextLayers);
    ui->comboBox_CAM_L3_2->clear();
    ui->comboBox_CAM_L4_2->clear();
    ui->comboBox_CAM_L5_2->clear();
    refreshVisInfo_2();
}

void ModelCAMPage::on_comboBox_L2_2(QString choicedLayer){
    this->choicedLayer_2["L2"] = choicedLayer.toStdString();
    this->choicedLayer_2["L3"] = "NULL";
    this->choicedLayer_2["L4"] = "NULL";
    this->choicedLayer_2["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L3(nextLayers, this->choicedLayer_2);
    ui->comboBox_CAM_L3_2->clear();
    ui->comboBox_CAM_L3_2->addItems(nextLayers);
    ui->comboBox_CAM_L4_2->clear();
    ui->comboBox_CAM_L5_2->clear();
    refreshVisInfo_2();
}

void ModelCAMPage::on_comboBox_L3_2(QString choicedLayer){
    this->choicedLayer_2["L3"] = choicedLayer.toStdString();
    this->choicedLayer_2["L4"] = "NULL";
    this->choicedLayer_2["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L4(nextLayers, this->choicedLayer_2);
    ui->comboBox_CAM_L4_2->clear();
    ui->comboBox_CAM_L4_2->addItems(nextLayers);
    ui->comboBox_CAM_L5_2->clear();
    refreshVisInfo_2();
}

void ModelCAMPage::on_comboBox_L4_2(QString choicedLayer){
    this->choicedLayer_2["L4"] = choicedLayer.toStdString();
    this->choicedLayer_2["L5"] = "NULL";

    QStringList nextLayers;
    loadModelStruct_L5(nextLayers, this->choicedLayer_2);
    ui->comboBox_CAM_L5_2->clear();
    ui->comboBox_CAM_L5_2->addItems(nextLayers);
    refreshVisInfo_2();
}

void ModelCAMPage::on_comboBox_L5_2(QString choicedLayer){
    this->choicedLayer_2["L5"] = choicedLayer.toStdString();
    refreshVisInfo_2();
}



// 样本选择下拉框相关槽接口
void ModelCAMPage::on_comboBox_stage_2(QString choicedStage){
    this->choicedStage_2 = choicedStage;
    // 扫描工程目录相应stage下的文件夹，将文件夹作为label下拉框的内容
    vector<string> labelNames = {};
    dirTools->getDirsplus(labelNames, (this->projectPath+"/"+this->choicedStage_2).toStdString());
    ui->comboBox_CAM_label_2->clear();
    ui->comboBox_CAM_label_2->addItems(CVS::fromStdVector(labelNames));
}

void ModelCAMPage::on_comboBox_label_2(QString choicedLabel){
    this->choicedLabel_2 = choicedLabel;
    // 扫描相应类别label目录下的.mat文件，将其作为mat下拉框的内容
    vector<string> matNames = {};
    dirTools->getFilesplus(matNames, ".mat", (this->projectPath+"/"+this->choicedStage_2+"/"+this->choicedLabel_2).toStdString());
    ui->comboBox_CAM_mat_2->clear();
    ui->comboBox_CAM_mat_2->addItems(CVS::fromStdVector(matNames));
}

void ModelCAMPage::on_comboBox_mat_2(QString choicedMat){
    this->choicedMatName_2 = choicedMat;
    this->choicedMatPATH_2 = this->projectPath+"/"+this->choicedStage_2+"/"+this->choicedLabel_2+"/"+choicedMat;
    if(dirTools->isExist(this->choicedMatPATH_2.toStdString())){
        MATFile* pMatFile = NULL;
        mxArray* pMxArray = NULL;
        pMatFile = matOpen(this->choicedMatPATH_2.toStdString().c_str(), "r");
        if(!pMatFile){qDebug()<<"(ModelEvnameOfMatFilealPage::takeSample)文件指针空!";return;}
        pMxArray = matGetNextVariable(pMatFile, NULL);
        if(!pMxArray){
            qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到!";
            return;
        }
        int N = mxGetN(pMxArray);  //N 列数
        // 给用户提示样本范围
        ui->lineEdit_CAM_begin_2->clear();
        ui->lineEdit_CAM_end_2->clear();
        ui->lineEdit_CAM_begin_2->setText("1");
        // 加历程图相关功能
        std::string dataType = projectsInfo->dataTypeOfSelectedProject;
        if(dataType == "IMAGE" || dataType == "RCS"){
            this->windowsLength = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsLength"]);
            this->windowsStep = QString::fromStdString(projectsInfo->getAllAttri(dataType,projectsInfo->nameOfSelectedProject)["Model_WindowsStep"]);
            if(this->windowsLength == "0" || this->windowsStep == "0", this->windowsLength == "" || this->windowsStep == ""){
                // 模型未训练，不能可视化
                QMessageBox::warning(NULL, "警告", "模型未训练，不能可视化！");
                return;
            }
            int sampleNum = (N - this->windowsLength.toInt())/this->windowsStep.toInt() + 1;
            ui->lineEdit_CAM_end_2->setText(QString::number(sampleNum));
            this->maxMatIndex_2 = sampleNum;
        }
        else{
            this->windowsLength = "0";
            this->windowsStep = "0";
            ui->lineEdit_CAM_end_2->setText(QString::number(N));
            this->maxMatIndex_2 = N;
        }
    }

}
/************************************************************************/
