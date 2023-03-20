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
    // 数据集类别选择框事件相应
    BtnGroup_typeChoice->addButton(ui->radioButton_train_choice, 0);
    BtnGroup_typeChoice->addButton(ui->radioButton_test_choice, 1);
    BtnGroup_typeChoice->addButton(ui->radioButton_val_choice, 2);
    BtnGroup_typeChoice->addButton(ui->radioButton_unknow_choice, 3);

    // connect(this->BtnGroup_typeChoice, &QButtonGroup::buttonClicked, this, &SenseSetPage::changeType);

    // 确定
    connect(ui->pushButton_datasetConfirm, &QPushButton::clicked, this, &SenseSetPage::confirmDataset);

    // 保存
//    connect(ui->pushButton_saveDatasetAttri, &QPushButton::clicked, this, &SenseSetPage::saveDatasetAttri);

    // 下一批数据
    connect(ui->pushButton_nextSenseChart, &QPushButton::clicked, this, &SenseSetPage::nextBatchChart);

    // 数据集属性显示框
    this->attriLabelGroup["Dataset_TargetNum"] = ui->label_sense_claNum;
    this->attriLabelGroup["Project_Path"] = ui->label_sense_PATH;
    this->attriLabelGroup["Dataset_Name"] = ui->label_sense_datasetName;
    this->attriLabelGroup["Dataset_TargetNumEachCla"] = ui->label_sense_targetNumEachCla;
    this->attriLabelGroup["Dataset_PitchAngle"] = ui->label_sense_pitchAngle;
    this->attriLabelGroup["Dataset_AzimuthAngle"] = ui->label_sense_azimuthAngle;
    this->attriLabelGroup["Dataset_SamplingNum"] = ui->label_sense_samplingNum;
    this->attriLabelGroup["Dataset_IncidentMode"] = ui->label_sense_incidentMode;
    this->attriLabelGroup["Dataset_Freq"] = ui->label_sense_freq;
    this->attriLabelGroup["Dataset_Note"] = ui->label_sense_note;

    // 图片显示label成组
    imgGroup.push_back(ui->label_datasetClaImg1);
    imgGroup.push_back(ui->label_datasetClaImg2);
    imgGroup.push_back(ui->label_datasetClaImg3);
    imgGroup.push_back(ui->label_datasetClaImg4);
    imgGroup.push_back(ui->label_datasetClaImg5);
    imgGroup.push_back(ui->label_datasetClaImg6);

    imgInfoGroup.push_back(ui->label_datasetCla1);
    imgInfoGroup.push_back(ui->label_datasetCla2);
    imgInfoGroup.push_back(ui->label_datasetCla3);
    imgInfoGroup.push_back(ui->label_datasetCla4);
    imgInfoGroup.push_back(ui->label_datasetCla5);
    imgInfoGroup.push_back(ui->label_datasetCla6);

    // 显示图表成组
    chartGroup.push_back(ui->label_senseChart1);
    chartGroup.push_back(ui->label_senseChart2);
    chartGroup.push_back(ui->label_senseChart3);
    chartGroup.push_back(ui->label_senseChart4);
    chartGroup.push_back(ui->label_senseChart5);
    chartGroup.push_back(ui->label_senseChart6);

    chartInfoGroup.push_back(ui->label_senseChartInfo_1);
    chartInfoGroup.push_back(ui->label_senseChartInfo_2);
    chartInfoGroup.push_back(ui->label_senseChartInfo_3);
    chartInfoGroup.push_back(ui->label_senseChartInfo_4);
    chartInfoGroup.push_back(ui->label_senseChartInfo_5);
    chartInfoGroup.push_back(ui->label_senseChartInfo_6);


}

SenseSetPage::~SenseSetPage(){

}

void SenseSetPage::confirmDataset(bool notDialog = false){
    QString project_path = QString::fromStdString(projectsInfo->pathOfSelectedProject);
    QString selectedType = this->BtnGroup_typeChoice->checkedButton()->objectName().split("_")[1];
    if(selectedType.isEmpty()||projectsInfo->pathOfSelectedProject==""){
        QMessageBox::warning(NULL, "数据集切换提醒", "数据集切换失败，活动工程或数据集未指定");
        return;
    }
    QString dataset_path = project_path + "/" + selectedType;
    bool ifDbExists = std::filesystem::exists(std::filesystem::u8path(dataset_path.toStdString()));
    if(!ifDbExists){
        QMessageBox::warning(NULL, "数据集切换提醒", "数据集切换失败，该工程下不存在"+selectedType+"数据集");
        return;
    }
    terminal->print("Selected Type: " + selectedType);

    projectsInfo->typeOfSelectedDataset = selectedType;
    projectsInfo->pathOfSelectedDataset = project_path.toStdString() + "/" + selectedType.toStdString();
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

    // 绘制类别图
    for(int i = 0; i<6; i++){
        imgGroup[i]->clear();
        imgInfoGroup[i]->clear();
    }
    drawClassImage();

    ui->progressBar->setValue(40);

    // 绘制曲线
    for(int i=0;i<6;i++){
        if(!chartGroup[i]->layout()) delete chartGroup[i]->layout();
        chartInfoGroup[i]->clear();
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
//    ui->plainTextEdit_sense_note->setPlainText(QString::fromStdString(attriContents["note"]));
}


void SenseSetPage::drawClassImage(){
    string rootPath = projectsInfo->pathOfSelectedDataset;
    vector<string> subDirNames = projectsInfo->classNamesOfSelectedDataset;
    for(int i = 0; i<subDirNames.size(); i++){
        imgInfoGroup[i]->setText(QString::fromStdString(subDirNames[i]));
        QString imgPath = QString::fromStdString(rootPath +"/"+ subDirNames[i] +".png");
        imgGroup[i]->setPixmap(QPixmap(imgPath).scaled(QSize(200,200), Qt::KeepAspectRatio));
    }
}


void SenseSetPage::nextBatchChart(){
    string rootPath = projectsInfo->pathOfSelectedDataset;
    vector<string> subDirNames = projectsInfo->classNamesOfSelectedDataset;
    // qDebug()<<"(SenseSetPage::nextBatchChart) subDirNames.size()="<<subDirNames.size();
    // 按类别显示
    for(int i=0; i<subDirNames.size(); i++){
        srand((unsigned)time(NULL));
        // 选取类别
        string choicedClass = subDirNames[i];
        string classPath = rootPath +"/"+ choicedClass;

        Chart *previewChart;

        vector<string> allMatFile;
        if(dirTools->getFilesplus(allMatFile, ".mat", classPath)){
            QString matFilePath = QString::fromStdString(classPath + "/" + allMatFile[0]);
            //下面这部分代码都是为了让randomIdx在合理的范围内（
            MATFile* pMatFile = NULL;
            mxArray* pMxArray = NULL;
            pMatFile = matOpen(matFilePath.toStdString().c_str(), "r");
            if(!pMatFile){qDebug()<<"(ModelEvalPage::randSample)文件指针空！！！！！！";return;}
            pMxArray = matGetNextVariable(pMatFile, NULL);
            if(!pMxArray){
                qDebug()<<"(Chart::readHRRPmat)pMxArray变量没找到！！！！！！";
                return;
            }
            int N = mxGetN(pMxArray);  //N 列数
            int randomIdx = N-(rand())%N;

            //绘图
            previewChart = new Chart(ui->label_mE_chartGT,QString::fromStdString(projectsInfo->dataTypeOfSelectedProject),matFilePath);
            previewChart->drawImage(chartGroup[i],randomIdx);
            chartInfoGroup[i]->setText(QString::fromStdString(choicedClass+":Index")+QString::number(randomIdx));
        }
    }
}
